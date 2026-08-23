FROM python:3.11-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build g++ nlohmann-json3-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY --from=build /app/kiln_py /app/kiln_py
WORKDIR /app

EXPOSE 8420
CMD ["uvicorn", "kiln_py.api.app:app", "--host", "0.0.0.0", "--port", "8420"]
