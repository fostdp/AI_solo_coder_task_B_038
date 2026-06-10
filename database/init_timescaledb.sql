-- 生物制药冻干机监控系统 TimescaleDB 初始化脚本
-- PostgreSQL 14 + TimescaleDB 2.11

-- 启用TimescaleDB扩展
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ========== 设备表 ==========
CREATE TABLE IF NOT EXISTS devices (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    location VARCHAR(100),
    status VARCHAR(20) DEFAULT 'running',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ========== 搁板表 ==========
CREATE TABLE IF NOT EXISTS shelves (
    id SERIAL PRIMARY KEY,
    device_id INTEGER REFERENCES devices(id) ON DELETE CASCADE,
    shelf_number INTEGER NOT NULL,
    temp_sensor_count INTEGER DEFAULT 8,
    vacuum_sensor_count INTEGER DEFAULT 2,
    UNIQUE(device_id, shelf_number)
);

-- ========== 遥测数据表 (超表) ==========
CREATE TABLE IF NOT EXISTS telemetry (
    timestamp TIMESTAMPTZ NOT NULL,
    device_id INTEGER NOT NULL,
    shelf_id INTEGER NOT NULL,
    temp_1 FLOAT, temp_2 FLOAT, temp_3 FLOAT, temp_4 FLOAT,
    temp_5 FLOAT, temp_6 FLOAT, temp_7 FLOAT, temp_8 FLOAT,
    vacuum_1 FLOAT, vacuum_2 FLOAT,
    cold_trap_temp FLOAT,
    power_1 FLOAT, power_2 FLOAT, power_3 FLOAT, power_4 FLOAT,
    power_5 FLOAT, power_6 FLOAT, power_7 FLOAT, power_8 FLOAT,
    PRIMARY KEY (timestamp, device_id, shelf_id),
    FOREIGN KEY (device_id) REFERENCES devices(id) ON DELETE CASCADE,
    FOREIGN KEY (shelf_id) REFERENCES shelves(id) ON DELETE CASCADE
);

-- 创建超表 (仅在表为空时执行)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables 
        WHERE hypertable_name = 'telemetry'
    ) THEN
        PERFORM create_hypertable('telemetry', 'timestamp');
    END IF;
END $$;

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_telemetry_device_time ON telemetry (device_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_telemetry_shelf_time ON telemetry (shelf_id, timestamp DESC);

-- 创建连续聚合视图：每分钟温度统计
CREATE MATERIALIZED VIEW IF NOT EXISTS telemetry_minute
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', timestamp) AS bucket,
    device_id,
    shelf_id,
    AVG((temp_1+temp_2+temp_3+temp_4+temp_5+temp_6+temp_7+temp_8)/8) AS avg_temp,
    MAX(GREATEST(temp_1,temp_2,temp_3,temp_4,temp_5,temp_6,temp_7,temp_8)) AS max_temp,
    MIN(LEAST(temp_1,temp_2,temp_3,temp_4,temp_5,temp_6,temp_7,temp_8)) AS min_temp,
    MAX(GREATEST(temp_1,temp_2,temp_3,temp_4,temp_5,temp_6,temp_7,temp_8)) - 
    MIN(LEAST(temp_1,temp_2,temp_3,temp_4,temp_5,temp_6,temp_7,temp_8)) AS temp_diff,
    AVG((vacuum_1+vacuum_2)/2) AS avg_vacuum,
    AVG(cold_trap_temp) AS avg_cold_trap
FROM telemetry
GROUP BY bucket, device_id, shelf_id
WITH NO DATA;

-- ========== 控制指令表 ==========
CREATE TABLE IF NOT EXISTS control_commands (
    id SERIAL PRIMARY KEY,
    device_id INTEGER REFERENCES devices(id) ON DELETE CASCADE,
    shelf_id INTEGER REFERENCES shelves(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    power_adj_1 FLOAT, power_adj_2 FLOAT, power_adj_3 FLOAT, power_adj_4 FLOAT,
    power_adj_5 FLOAT, power_adj_6 FLOAT, power_adj_7 FLOAT, power_adj_8 FLOAT,
    auto_mode BOOLEAN DEFAULT true
);

CREATE INDEX IF NOT EXISTS idx_control_device_time ON control_commands (device_id, timestamp DESC);

-- ========== 预测结果表 ==========
CREATE TABLE IF NOT EXISTS prediction_results (
    id SERIAL PRIMARY KEY,
    device_id INTEGER REFERENCES devices(id) ON DELETE CASCADE,
    batch_id VARCHAR(50),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    moisture_pred FLOAT,
    moisture_conf FLOAT,
    moisture_threshold FLOAT DEFAULT 3.0,
    reconstitution_pred FLOAT,
    reconstitution_conf FLOAT,
    reconstitution_threshold FLOAT DEFAULT 120.0,
    drying_rate FLOAT,
    is_qualified BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_prediction_device_time ON prediction_results (device_id, timestamp DESC);

-- ========== 告警表 ==========
CREATE TABLE IF NOT EXISTS alarms (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    device_id INTEGER REFERENCES devices(id) ON DELETE CASCADE,
    shelf_id INTEGER REFERENCES shelves(id) ON DELETE CASCADE,
    alarm_type VARCHAR(30) NOT NULL,
    severity VARCHAR(10) NOT NULL,
    message TEXT NOT NULL,
    acknowledged BOOLEAN DEFAULT false,
    acknowledged_by VARCHAR(50),
    acknowledged_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_alarm_device_time ON alarms (device_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alarm_acknowledged ON alarms (acknowledged) WHERE acknowledged = false;

-- ========== 系统配置表 ==========
CREATE TABLE IF NOT EXISTS system_config (
    key VARCHAR(50) PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 插入默认配置
INSERT INTO system_config (key, value) VALUES
('temp_diff_threshold', '1.0'),
('vacuum_min_threshold', '0.1'),
('vacuum_max_threshold', '100.0'),
('cold_trap_max_threshold', '-50.0'),
('moisture_max_threshold', '3.0'),
('reconstitution_max_threshold', '120.0'),
('control_interval', '10'),
('mqtt_broker', 'localhost'),
('mqtt_port', '1883'),
('mqtt_topic', 'pharmacy/mes/alarm'),
('auto_control_enabled', 'true')
ON CONFLICT (key) DO NOTHING;

-- ========== 初始化设备数据 ==========
INSERT INTO devices (name, location) VALUES
('FD-001', '车间A-1号'), ('FD-002', '车间A-2号'), ('FD-003', '车间A-3号'),
('FD-004', '车间B-1号'), ('FD-005', '车间B-2号'), ('FD-006', '车间B-3号'),
('FD-007', '车间C-1号'), ('FD-008', '车间C-2号'), ('FD-009', '车间C-3号'),
('FD-010', '车间D-1号')
ON CONFLICT DO NOTHING;

-- 初始化搁板数据
DO $$
DECLARE
    d_id INTEGER;
    s_num INTEGER;
BEGIN
    FOR d_id IN 1..10 LOOP
        FOR s_num IN 1..5 LOOP
            INSERT INTO shelves (device_id, shelf_number) 
            VALUES (d_id, s_num)
            ON CONFLICT (device_id, shelf_number) DO NOTHING;
        END LOOP;
    END LOOP;
END $$;

-- ========== 查询示例 ==========
-- ========== 冻干终点判定表 ==========
CREATE TABLE IF NOT EXISTS drying_endpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id INTEGER REFERENCES devices(id) ON DELETE CASCADE,
    batch_id VARCHAR(50) NOT NULL,
    cycle_phase VARCHAR(20) NOT NULL,  -- primary_drying, secondary_drying
    detection_method VARCHAR(30) NOT NULL,  -- first_derivative, autoencoder, pressure_rise_test
    endpoint_timestamp TIMESTAMPTZ NOT NULL,
    detection_confidence FLOAT,
    pressure_rise_delta FLOAT,  -- 压力升测试的压力变化
    temp_inflection_point FLOAT,  -- 温度拐点值
    temp_first_derivative FLOAT,  -- 一阶导数值
    autoencoder_recon_error FLOAT,  -- 自编码器重构误差
    cycle_duration_hours FLOAT,  -- 该阶段持续时间
    estimated_energy_saving FLOAT,  -- 预计节省能耗
    is_accepted BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_endpoint_device_time ON drying_endpoints (device_id, endpoint_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_endpoint_batch ON drying_endpoints (batch_id);

-- ========== 冷阱除霜优化表 ==========
CREATE TABLE IF NOT EXISTS defrost_optimizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id INTEGER REFERENCES devices(id) ON DELETE CASCADE,
    batch_id VARCHAR(50),
    timestamp TIMESTAMPTZ NOT NULL,
    estimated_frost_thickness_mm FLOAT,  -- 估算结霜厚度
    cold_trap_temp_avg FLOAT,  -- 冷阱平均温度
    cold_trap_temp_trend FLOAT,  -- 温度趋势斜率
    recommended_defrost_interval_hours FLOAT,  -- 推荐除霜间隔
    recommended_heating_power_pct FLOAT,  -- 推荐加热功率百分比
    actual_defrost_start_time TIMESTAMPTZ,
    actual_defrost_end_time TIMESTAMPTZ,
    actual_energy_consumed_kwh FLOAT,  -- 实际能耗
    estimated_energy_saving FLOAT,  -- 预计节能
    defrost_status VARCHAR(20) DEFAULT 'pending',  -- pending, in_progress, completed, skipped
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_defrost_device_time ON defrost_optimizations (device_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_defrost_status ON defrost_optimizations (defrost_status);

-- ========== 电价时间表 ==========
CREATE TABLE IF NOT EXISTS electricity_prices (
    id SERIAL PRIMARY KEY,
    price_date DATE NOT NULL,
    hour_of_day INTEGER NOT NULL,
    price_cny_per_kwh FLOAT NOT NULL,  -- 电价（元/度）
    price_type VARCHAR(20) DEFAULT 'standard',  -- peak, flat, valley, standard
    is_holiday BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(price_date, hour_of_day)
);

CREATE INDEX IF NOT EXISTS idx_electricity_date ON electricity_prices (price_date);

-- ========== 群控调度表 ==========
CREATE TABLE IF NOT EXISTS fleet_schedules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    schedule_date DATE NOT NULL,
    total_required_batches INTEGER NOT NULL,
    estimated_energy_cost FLOAT,  -- 预计电费
    optimized_energy_saving FLOAT,  -- 优化后节省
    solver_status VARCHAR(20),  -- optimal, suboptimal, timeout
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_schedule_date ON fleet_schedules (schedule_date DESC);

-- ========== 群控调度详情表 ==========
CREATE TABLE IF NOT EXISTS fleet_schedule_details (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    schedule_id UUID REFERENCES fleet_schedules(id) ON DELETE CASCADE,
    device_id INTEGER REFERENCES devices(id) ON DELETE CASCADE,
    batch_id VARCHAR(50),
    formula_id VARCHAR(50),
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    freeze_profile_id INTEGER,  -- 冻干曲线ID
    estimated_cycle_hours FLOAT,
    estimated_energy_kwh FLOAT,
    priority INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_schedule_detail_schedule ON fleet_schedule_details (schedule_id);
CREATE INDEX IF NOT EXISTS idx_schedule_detail_device ON fleet_schedule_details (device_id);

-- ========== 冻干曲线配方表 ==========
CREATE TABLE IF NOT EXISTS freeze_profiles (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    formula_id VARCHAR(50) NOT NULL,
    description TEXT,
    primary_drying_temp FLOAT NOT NULL,  -- 一次干燥温度
    primary_drying_pressure FLOAT NOT NULL,  -- 一次干燥压力
    primary_drying_duration_hours FLOAT,  -- 预计一次干燥时间
    secondary_drying_temp FLOAT NOT NULL,  -- 二次干燥温度
    secondary_drying_pressure FLOAT NOT NULL,  -- 二次干燥压力
    secondary_drying_duration_hours FLOAT,  -- 预计二次干燥时间
    estimated_energy_kwh FLOAT,  -- 预计能耗
    estimated_cycle_hours FLOAT,  -- 预计总周期
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(formula_id)
);

-- ========== 制品缺陷检测表 ==========
CREATE TABLE IF NOT EXISTS product_defects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id INTEGER REFERENCES devices(id) ON DELETE CASCADE,
    batch_id VARCHAR(50) NOT NULL,
    shelf_id INTEGER REFERENCES shelves(id),
    vial_position VARCHAR(20),  -- 西林瓶位置
    image_path VARCHAR(255),  -- 图像存储路径
    image_hash VARCHAR(64),  -- 图像哈希值用于去重
    defect_type VARCHAR(30) NOT NULL,  -- collapse, atrophy, cracking, normal
    defect_severity VARCHAR(20) DEFAULT 'low',  -- low, medium, high
    confidence FLOAT NOT NULL,  -- 分类置信度
    bbox_x INTEGER,  -- 缺陷边界框
    bbox_y INTEGER,
    bbox_width INTEGER,
    bbox_height INTEGER,
    is_manual_reviewed BOOLEAN DEFAULT false,
    manual_label VARCHAR(30),
    reviewed_by VARCHAR(50),
    reviewed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_defect_device_time ON product_defects (device_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_defect_batch ON product_defects (batch_id);
CREATE INDEX IF NOT EXISTS idx_defect_type ON product_defects (defect_type);

-- ========== 批次记录扩展表 ==========
CREATE TABLE IF NOT EXISTS batch_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id INTEGER REFERENCES devices(id) ON DELETE CASCADE,
    batch_id VARCHAR(50) NOT NULL UNIQUE,
    formula_id VARCHAR(50),
    freeze_profile_id INTEGER REFERENCES freeze_profiles(id),
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    primary_drying_endpoint TIMESTAMPTZ,
    secondary_drying_endpoint TIMESTAMPTZ,
    total_cycle_hours FLOAT,
    total_energy_kwh FLOAT,
    avg_moisture_content FLOAT,
    avg_reconstitution_time FLOAT,
    defect_rate FLOAT,  -- 缺陷率
    total_vials_count INTEGER,  -- 总瓶数
    defective_vials_count INTEGER,  -- 缺陷瓶数
    batch_status VARCHAR(20) DEFAULT 'running',  -- running, completed, aborted
    quality_score FLOAT,
    operator VARCHAR(50),
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_batch_device_time ON batch_records (device_id, start_time DESC);
CREATE INDEX IF NOT EXISTS idx_batch_status ON batch_records (batch_status);

-- ========== 压力升测试记录表 ==========
CREATE TABLE IF NOT EXISTS pressure_rise_tests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id INTEGER REFERENCES devices(id) ON DELETE CASCADE,
    batch_id VARCHAR(50),
    test_start_time TIMESTAMPTZ NOT NULL,
    test_end_time TIMESTAMPTZ,
    initial_pressure_pa FLOAT,  -- 初始压力
    final_pressure_pa FLOAT,  -- 结束压力
    pressure_rise_pa_per_min FLOAT,  -- 压力升速率
    test_duration_seconds INTEGER,
    is_endpoint_detected BOOLEAN DEFAULT false,
    detection_confidence FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prt_device_time ON pressure_rise_tests (device_id, test_start_time DESC);
CREATE INDEX IF NOT EXISTS idx_prt_batch ON pressure_rise_tests (batch_id);

-- ========== 初始化冻干曲线配方数据 ==========
INSERT INTO freeze_profiles (name, formula_id, description, 
    primary_drying_temp, primary_drying_pressure, primary_drying_duration_hours,
    secondary_drying_temp, secondary_drying_pressure, secondary_drying_duration_hours,
    estimated_energy_kwh, estimated_cycle_hours) VALUES
('标准配方A', 'FORMULA-001', '标准蛋白类产品冻干曲线', -45.0, 0.1, 24.0, 25.0, 0.05, 8.0, 120.0, 32.0),
('快速配方B', 'FORMULA-002', '节能快速冻干曲线', -40.0, 0.15, 18.0, 30.0, 0.03, 6.0, 100.0, 24.0),
('温和配方C', 'FORMULA-003', '热敏性产品温和曲线', -50.0, 0.08, 30.0, 20.0, 0.05, 12.0, 150.0, 42.0),
('高真空配方D', 'FORMULA-004', '高真空精密冻干曲线', -48.0, 0.05, 28.0, 22.0, 0.02, 10.0, 140.0, 38.0)
ON CONFLICT (formula_id) DO NOTHING;

-- ========== 初始化电价数据（示例） ==========
INSERT INTO electricity_prices (price_date, hour_of_day, price_cny_per_kwh, price_type)
SELECT 
    CURRENT_DATE + (i::INTEGER / 24) * INTERVAL '1 day',
    i % 24,
    CASE 
        WHEN (i % 24) BETWEEN 7 AND 10 OR (i % 24) BETWEEN 18 AND 23 THEN 1.2  -- 峰时
        WHEN (i % 24) BETWEEN 11 AND 17 THEN 0.8  -- 平时
        ELSE 0.4  -- 谷时
    END,
    CASE 
        WHEN (i % 24) BETWEEN 7 AND 10 OR (i % 24) BETWEEN 18 AND 23 THEN 'peak'
        WHEN (i % 24) BETWEEN 11 AND 17 THEN 'flat'
        ELSE 'valley'
    END
FROM generate_series(0, 167) AS s(i)  -- 未来7天
ON CONFLICT (price_date, hour_of_day) DO NOTHING;

-- ========== 新增系统配置 ==========
INSERT INTO system_config (key, value) VALUES
('endpoint_detection_enabled', 'true'),
('defrost_optimization_enabled', 'true'),
('fleet_control_enabled', 'true'),
('defect_detection_enabled', 'true'),
('primary_drying_temp_threshold', '-10.0'),
('secondary_drying_temp_threshold', '20.0'),
('pressure_rise_threshold', '0.05'),
('autoencoder_threshold', '0.1'),
('frost_thickness_max_mm', '5.0'),
('defrost_power_max_pct', '80.0'),
('max_defrost_duration_minutes', '60')
ON CONFLICT (key) DO NOTHING;

-- ========== 查询示例 ==========
-- 查询最新实时数据
-- SELECT * FROM telemetry 
-- WHERE device_id = 1 
-- ORDER BY timestamp DESC 
-- LIMIT 50;

-- 查询温度统计
-- SELECT * FROM telemetry_minute 
-- WHERE device_id = 1 AND bucket > NOW() - INTERVAL '1 hour'
-- ORDER BY bucket DESC;

-- 查询批次记录
-- SELECT * FROM batch_records WHERE batch_status = 'running';

-- 查询缺陷统计
-- SELECT batch_id, defect_type, COUNT(*) FROM product_defects 
-- WHERE batch_id = 'BATCH-001' GROUP BY batch_id, defect_type;
