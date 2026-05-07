import asyncio

from app.storage import models as _models
from app.storage.database import create_db_schema


async def main() -> None:
    await create_db_schema()


if __name__ == "__main__":
    asyncio.run(main())
