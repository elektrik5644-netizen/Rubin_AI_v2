# 🧠 Анализ интеллектуального диспетчера LocalAI

## 📊 Общая архитектура

LocalAI использует многоуровневую систему диспетчеризации с несколькими ключевыми компонентами:

### 1. **FederatedServer - Основной диспетчер**
```go
type FederatedServer struct {
    sync.Mutex
    listenAddr, service, p2ptoken string
    requestTable                  map[string]int  // Счетчик запросов по нодам
    loadBalanced                  bool            // Режим балансировки нагрузки
    workerTarget                  string          // Целевой воркер
}
```

### 2. **ModelLoader - Загрузчик и менеджер моделей**
```go
type ModelLoader struct {
    ModelPath        string
    mu               sync.Mutex
    singletonMode    bool              // Режим одной активной модели
    models           map[string]*Model // Загруженные модели
    wd               *WatchDog         // Мониторинг процессов
    externalBackends map[string]string // Внешние бэкенды
}
```

### 3. **BackendMonitorService - Мониторинг производительности**
```go
type BackendMonitorService struct {
    modelConfigLoader *config.ModelConfigLoader
    modelLoader       *model.ModelLoader
    options           *config.ApplicationConfig
}
```

## 🔄 Алгоритмы диспетчеризации

### **1. Балансировка нагрузки (Load Balancing)**

#### **SelectLeastUsedServer() - Выбор наименее загруженного сервера:**
```go
func (fs *FederatedServer) SelectLeastUsedServer() string {
    fs.syncTableStatus()
    
    fs.Lock()
    defer fs.Unlock()
    
    // Поиск ноды с минимальным количеством запросов
    var min int
    var minKey string
    for k, v := range fs.requestTable {
        if min == 0 || v < min {
            min = v
            minKey = k
        }
    }
    
    return minKey
}
```

#### **RandomServer() - Случайный выбор:**
```go
func (fs *FederatedServer) RandomServer() string {
    var tunnelAddresses []string
    for _, v := range GetAvailableNodes(fs.service) {
        if v.IsOnline() {
            tunnelAddresses = append(tunnelAddresses, v.ID)
        } else {
            delete(fs.requestTable, v.ID) // Очистка оффлайн нод
        }
    }
    
    if len(tunnelAddresses) == 0 {
        return ""
    }
    
    return tunnelAddresses[rand.IntN(len(tunnelAddresses))]
}
```

### **2. Мониторинг состояния нод**

#### **Синхронизация таблицы состояний:**
```go
func (fs *FederatedServer) syncTableStatus() {
    fs.Lock()
    defer fs.Unlock()
    currentTunnels := make(map[string]struct{})
    
    // Обновление активных нод
    for _, v := range GetAvailableNodes(fs.service) {
        if v.IsOnline() {
            fs.ensureRecordExist(v.ID)
            currentTunnels[v.ID] = struct{}{}
        }
    }
    
    // Удаление неактивных нод
    for t := range fs.requestTable {
        if _, ok := currentTunnels[t]; !ok {
            delete(fs.requestTable, t)
        }
    }
}
```

### **3. Мониторинг производительности бэкендов**

#### **Отслеживание ресурсов:**
```go
func (bms *BackendMonitorService) SampleLocalBackendProcess(model string) (*schema.BackendMonitorResponse, error) {
    // Получение PID процесса
    pid, err := bms.modelLoader.GetGRPCPID(backend)
    
    // Создание объекта процесса
    backendProcess, err := gopsutil.NewProcess(int32(pid))
    
    // Сбор метрик
    memInfo, err := backendProcess.MemoryInfo()
    memPercent, err := backendProcess.MemoryPercent()
    cpuPercent, err := backendProcess.CPUPercent()
    
    return &schema.BackendMonitorResponse{
        MemoryInfo:    memInfo,
        MemoryPercent: memPercent,
        CPUPercent:    cpuPercent,
    }, nil
}
```

## 🎯 Стратегии маршрутизации

### **1. Режимы работы диспетчера:**

#### **Load Balanced Mode:**
```go
if fs.loadBalanced {
    log.Debug().Msgf("Load balancing request")
    
    workerID = fs.SelectLeastUsedServer()
    if workerID == "" {
        log.Debug().Msgf("Least used server not found, selecting random")
        workerID = fs.RandomServer()
    }
} else {
    workerID = fs.RandomServer()
}
```

#### **Targeted Mode:**
```go
if fs.workerTarget != "" {
    workerID = fs.workerTarget  // Прямая маршрутизация к конкретному воркеру
}
```

### **2. Обработка ошибок и fallback:**

```go
if workerID == "" {
    log.Error().Msg("No available nodes yet")
    fs.sendHTMLResponse(conn, 503, "Sorry, waiting for nodes to connect")
    return
}

nodeData, exists := GetNode(fs.service, workerID)
if !exists {
    log.Error().Msgf("Node %s not found", workerID)
    fs.sendHTMLResponse(conn, 404, "Node not found")
    return
}
```

## 📈 Метрики и мониторинг

### **1. Счетчик запросов:**
```go
func (fs *FederatedServer) RecordRequest(nodeID string) {
    fs.Lock()
    defer fs.Unlock()
    fs.requestTable[nodeID]++  // Инкремент счетчика
    
    log.Debug().Any("request_table", fs.requestTable).Any("request", nodeID).Msgf("Recording request")
}
```

### **2. Мониторинг состояния бэкендов:**
```go
func (bms BackendMonitorService) CheckAndSample(modelName string) (*proto.StatusResponse, error) {
    modelAddr := bms.modelLoader.CheckIsLoaded(modelName)
    if modelAddr == nil {
        return nil, fmt.Errorf("backend %s is not currently loaded", modelName)
    }
    
    // Проверка через gRPC
    status, rpcErr := modelAddr.GRPC(false, nil).Status(context.TODO())
    if rpcErr != nil {
        // Fallback на локальный мониторинг процесса
        val, slbErr := bms.SampleLocalBackendProcess(modelName)
        // ...
    }
    
    return status, nil
}
```

## 🔧 Ключевые принципы

### **1. Отказоустойчивость:**
- Автоматическое удаление оффлайн нод
- Fallback на случайный выбор при недоступности
- Мониторинг состояния процессов

### **2. Масштабируемость:**
- P2P архитектура для распределения нагрузки
- Динамическое добавление/удаление нод
- Балансировка на основе реальной нагрузки

### **3. Мониторинг:**
- Отслеживание ресурсов (CPU, память)
- Счетчики запросов по нодам
- gRPC health checks

### **4. Гибкость:**
- Различные режимы маршрутизации
- Поддержка внешних бэкендов
- Конфигурируемые стратегии

## 🚀 Применение в Rubin AI

### **Адаптация принципов LocalAI для Rubin AI:**

#### **1. Интеллектуальная маршрутизация:**
```python
class IntelligentDispatcher:
    def __init__(self):
        self.request_table = {}  # Счетчик запросов по модулям
        self.module_health = {}  # Состояние модулей
        self.load_balanced = True
    
    def select_best_module(self, category):
        """Выбор наименее загруженного модуля"""
        if self.load_balanced:
            return self.select_least_used_module(category)
        else:
            return self.random_module(category)
    
    def select_least_used_module(self, category):
        """Выбор модуля с минимальной нагрузкой"""
        available_modules = self.get_available_modules(category)
        if not available_modules:
            return None
            
        min_requests = min(self.request_table.get(module, 0) for module in available_modules)
        least_used = [m for m in available_modules if self.request_table.get(m, 0) == min_requests]
        
        return random.choice(least_used) if least_used else None
```

#### **2. Мониторинг производительности:**
```python
class ModuleMonitor:
    def __init__(self):
        self.module_metrics = {}
    
    def sample_module_performance(self, module_name):
        """Сбор метрик производительности модуля"""
        try:
            response = requests.get(f"http://localhost:{self.get_module_port(module_name)}/health", timeout=3)
            if response.status_code == 200:
                return {
                    'status': 'online',
                    'response_time': response.elapsed.total_seconds(),
                    'timestamp': datetime.now()
                }
        except:
            return {'status': 'offline', 'timestamp': datetime.now()}
    
    def update_module_health(self, module_name):
        """Обновление состояния модуля"""
        self.module_metrics[module_name] = self.sample_module_performance(module_name)
```

#### **3. Адаптивная балансировка:**
```python
class AdaptiveLoadBalancer:
    def __init__(self):
        self.dispatcher = IntelligentDispatcher()
        self.monitor = ModuleMonitor()
        self.performance_history = {}
    
    def route_request(self, request_data):
        """Маршрутизация запроса с учетом производительности"""
        category = self.analyze_request_category(request_data)
        
        # Выбор модуля на основе текущей нагрузки и производительности
        best_module = self.select_optimal_module(category)
        
        if best_module:
            # Запись запроса
            self.dispatcher.record_request(best_module)
            
            # Маршрутизация
            return self.forward_request(best_module, request_data)
        else:
            return self.handle_no_available_modules()
    
    def select_optimal_module(self, category):
        """Выбор оптимального модуля на основе метрик"""
        available_modules = self.get_available_modules(category)
        
        if not available_modules:
            return None
        
        # Сортировка по комбинации нагрузки и производительности
        scored_modules = []
        for module in available_modules:
            load_score = self.dispatcher.request_table.get(module, 0)
            perf_score = self.calculate_performance_score(module)
            total_score = load_score + perf_score
            
            scored_modules.append((module, total_score))
        
        # Выбор модуля с наименьшим score
        scored_modules.sort(key=lambda x: x[1])
        return scored_modules[0][0] if scored_modules else None
```

## 🎯 Заключение

**LocalAI использует интеллектуальный диспетчер с следующими ключевыми особенностями:**

1. **Многоуровневая архитектура** - FederatedServer, ModelLoader, BackendMonitor
2. **Адаптивная балансировка** - выбор наименее загруженных нод
3. **Мониторинг в реальном времени** - отслеживание ресурсов и состояния
4. **Отказоустойчивость** - автоматическое восстановление и fallback
5. **P2P распределение** - масштабируемость через сеть нод

**Для Rubin AI можно адаптировать:**
- Систему счетчиков запросов по модулям
- Мониторинг производительности специализированных API
- Адаптивную маршрутизацию на основе нагрузки
- Health checks и автоматическое восстановление

**Это позволит создать интеллектуальный диспетчер, который будет оптимально распределять запросы между специализированными модулями Rubin AI.**

















