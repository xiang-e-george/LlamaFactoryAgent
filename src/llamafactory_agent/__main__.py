"""本地 API 的命令行入口。"""

import uvicorn

from .api import create_app
from .config import Settings


def main() -> None:
    """以单进程启动 API，避免多个调度器重复领取任务。

    @return: 服务停止后返回。
    """
    settings = Settings()
    uvicorn.run(create_app(settings), host=settings.api_host, port=settings.api_port, workers=1)


if __name__ == "__main__":
    main()
