"""
Мониторинг и метрики для Rubin AI Matrix Gateway
"""

from fastapi import FastAPI
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
import logging

logger = logging.getLogger(__name__)

# Метрики Prometheus
REQUEST_COUNT = Counter(
    'rubin_gateway_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'rubin_gateway_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'rubin_gateway_active_connections',
    'Number of active connections'
)

MATRIX_TASKS_TOTAL = Counter(
    'rubin_matrix_tasks_total',
    'Total number of matrix tasks',
    ['task_type', 'status']
)

NODE_RESPONSE_TIME = Histogram(
    'rubin_node_response_time_seconds',
    'Response time from matrix nodes',
    ['node_name', 'operation']
)

def setup_monitoring(app: FastAPI):
    """Настройка мониторинга для приложения"""
    
    @app.middleware("http")
    async def monitor_requests(request, call_next):
        """Middleware для мониторинга запросов"""
        start_time = time.time()
        
        # Увеличение счетчика активных соединений
        ACTIVE_CONNECTIONS.inc()
        
        try:
            response = await call_next(request)
            
            # Запись метрик
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(time.time() - start_time)
            
            return response
            
        except Exception as e:
            # Запись ошибки
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=500
            ).inc()
            
            logger.error(f"Request monitoring error: {e}")
            raise
            
        finally:
            # Уменьшение счетчика активных соединений
            ACTIVE_CONNECTIONS.dec()
    
    @app.get("/metrics")
    async def metrics():
        """Эндпоинт для метрик Prometheus"""
        return Response(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    logger.info("Monitoring setup completed")

def record_matrix_task(task_type: str, status: str):
    """Запись метрики задачи матрицы"""
    MATRIX_TASKS_TOTAL.labels(
        task_type=task_type,
        status=status
    ).inc()

def record_node_response_time(node_name: str, operation: str, duration: float):
    """Запись времени ответа узла"""
    NODE_RESPONSE_TIME.labels(
        node_name=node_name,
        operation=operation
    ).observe(duration)
