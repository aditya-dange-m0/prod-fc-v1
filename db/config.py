import os
import ssl
import uuid
from functools import lru_cache
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Simple database configuration - just what you need."""

    # CONNECTION URLS (Required)
    DATABASE_URL: str
    DIRECT_DATABASE_URL: str

    # CONNECTION POOL SETTINGS
    POOL_SIZE: int = 10
    MAX_OVERFLOW: int = 20
    POOL_TIMEOUT: float = 30.0
    POOL_RECYCLE: int = 3600
    POOL_PRE_PING: bool = True

    # TIMEOUTS
    CONNECT_TIMEOUT: int = 10
    COMMAND_TIMEOUT: int = 30

    # LOGGING
    ECHO_SQL: bool = False
    ECHO_POOL: bool = False

    # ENVIRONMENT
    ENV: str = "development"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env

    def get_connection_url(self, use_direct: bool = False) -> str:
        """
        Get connection URL with asyncpg dialect enforced.
        Removes sslmode parameter since asyncpg handles SSL via connect_args.

        Args:
            use_direct: If True, use direct connection
                       If False, use pooled connection

        Returns:
            Connection URL with postgresql+asyncpg:// dialect and sslmode removed

        Raises:
            ValueError: If URL format is invalid
        """
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

        url = self.DIRECT_DATABASE_URL if use_direct else self.DATABASE_URL

        # ✅ FIX: Ensure asyncpg dialect is always present
        # This prevents SQLAlchemy from defaulting to psycopg2
        if url.startswith("postgresql://"):
            # Replace plain postgresql:// with postgresql+asyncpg://
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql+asyncpg://"):
            # Already correct, do nothing
            pass
        else:
            raise ValueError(
                f"Invalid database URL format. "
                f"Expected 'postgresql://' or 'postgresql+asyncpg://', "
                f"got: {url[:30]}..."
            )

        # ✅ FIX: Remove sslmode and channel_binding from URL
        # asyncpg doesn't accept these as URL parameters, we handle SSL in connect_args
        parsed = urlparse(url)
        if parsed.query:
            query_params = parse_qs(parsed.query)
            # Store sslmode for later use in get_connect_args
            self._parsed_sslmode = query_params.get("sslmode", ["prefer"])[0]
            # Remove SSL-related parameters that asyncpg doesn't understand
            query_params.pop("sslmode", None)
            query_params.pop("channel_binding", None)
            # Rebuild query string
            new_query = urlencode(query_params, doseq=True)
            url = urlunparse(parsed._replace(query=new_query))
        else:
            self._parsed_sslmode = "prefer"

        return url

    def get_connect_args(self) -> dict:
        """
        Get asyncpg connection arguments with SSL.

        ✅ FIX: Handle SSL based on parsed sslmode from URL or SSL_MODE env var
        Supports Neon DB which requires SSL

        Returns:
            dict: Connection arguments for asyncpg
        """
        # Get SSL mode from parsed URL or environment variable
        ssl_mode = getattr(
            self, "_parsed_sslmode", os.getenv("SSL_MODE", "prefer")
        ).lower()

        connect_args = {
            # Prevent prepared statement conflicts with connection poolers
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0,
            "prepared_statement_name_func": lambda: f"__asyncpg_{uuid.uuid4().hex[:16]}__",
            # Timeouts
            "timeout": self.CONNECT_TIMEOUT,
            "command_timeout": self.COMMAND_TIMEOUT,
            # Server settings
            "server_settings": {
                "application_name": f"app_{self.ENV}",
                "statement_timeout": str(self.COMMAND_TIMEOUT * 1000),
            },
        }

        # Handle SSL based on sslmode
        if ssl_mode == "disable":
            # No SSL
            connect_args["ssl"] = False
        elif ssl_mode == "require":
            # SSL required - for Neon DB
            # Create SSL context that requires SSL but doesn't verify certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            connect_args["ssl"] = ssl_context
        elif ssl_mode == "prefer":
            # SSL preferred but not required
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            connect_args["ssl"] = ssl_context
        else:
            # Default SSL behavior
            connect_args["ssl"] = True

        return connect_args

    def mask_sensitive_data(self) -> dict:
        """
        Return config with masked passwords for safe logging.

        Returns:
            dict: Configuration with passwords replaced by ***MASKED***
        """
        from urllib.parse import urlparse

        config = self.model_dump()

        for url_key in ["DATABASE_URL", "DIRECT_DATABASE_URL"]:
            if config.get(url_key):
                try:
                    parsed = urlparse(config[url_key])
                    if parsed.password:
                        config[url_key] = config[url_key].replace(
                            parsed.password, "***MASKED***"
                        )
                except Exception:
                    config[url_key] = "***MASKED***"

        return config

    def validate_urls(self) -> None:
        """
        Validate that URLs are properly formatted.

        Raises:
            ValueError: If URLs are missing or invalid
        """
        if not self.DATABASE_URL:
            raise ValueError("DATABASE_URL is required")
        if not self.DIRECT_DATABASE_URL:
            raise ValueError("DIRECT_DATABASE_URL is required")

        # Test that URLs can be processed
        try:
            self.get_connection_url(use_direct=False)
            self.get_connection_url(use_direct=True)
        except Exception as e:
            raise ValueError(f"Invalid database URL format: {e}")


@lru_cache()
def get_db_settings() -> DatabaseSettings:
    """
    Get cached database settings singleton.

    Returns:
        DatabaseSettings: Singleton instance of database settings
    """
    settings = DatabaseSettings()
    settings.validate_urls()  # Validate on first access
    return settings
