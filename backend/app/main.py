from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime, timedelta
from sqlalchemy import text
from app.core.config import settings
from app.core.database import engine, AsyncSessionLocal
from app.api import devices, data, control, prediction, alarm, endpoint, defrost, fleet, defect
from app.services.alarm import AlarmEvent
from app.api.alarm import alarm_detector, mqtt_service, process_new_alarms


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    await mqtt_service.start()
    
    control_task = asyncio.create_task(background_control_loop())
    prediction_task = asyncio.create_task(background_prediction_loop())
    alarm_task = asyncio.create_task(background_alarm_check_loop())
    
    yield
    
    control_task.cancel()
    prediction_task.cancel()
    alarm_task.cancel()
    
    await mqtt_service.stop()
    await engine.dispose()
    print("Application shutdown complete")


app = FastAPI(
    title=settings.APP_NAME,
    description="生物制药冻干机搁板温度均匀性控制与产品质量预测系统",
    version=settings.APP_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(devices.router)
app.include_router(data.router)
app.include_router(control.router)
app.include_router(prediction.router)
app.include_router(alarm.router)
app.include_router(fleet.router)
app.include_router(endpoint.router)
app.include_router(defrost.router)
app.include_router(defect.router)


async def background_control_loop():
    while True:
        try:
            if settings.AUTO_CONTROL_ENABLED:
                for device_id in range(1, 11):
                    async with AsyncSessionLocal() as db:
                        for shelf_id in range(1, 6):
                            try:
                                query = text(f"""
                                    SELECT DISTINCT ON (shelf_id)
                                        temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8,
                                        power_1, power_2, power_3, power_4, power_5, power_6, power_7, power_8
                                    FROM telemetry
                                    WHERE device_id = :device_id AND shelf_id = :shelf_id
                                      AND timestamp >= :start_time
                                    ORDER BY shelf_id, timestamp DESC
                                """)
                                
                                start_time = datetime.now() - timedelta(minutes=5)
                                result = await db.execute(query, {
                                    "device_id": device_id,
                                    "shelf_id": shelf_id,
                                    "start_time": start_time
                                })
                                row = result.first()
                                
                                if row:
                                    from app.api.control import controller, auto_mode
                                    
                                    if auto_mode.get(device_id, True):
                                        temperatures = [row.temp_1, row.temp_2, row.temp_3, row.temp_4,
                                                       row.temp_5, row.temp_6, row.temp_7, row.temp_8]
                                        powers = [row.power_1, row.power_2, row.power_3, row.power_4,
                                                 row.power_5, row.power_6, row.power_7, row.power_8]
                                        
                                        temperatures = [t for t in temperatures if t is not None]
                                        powers = [p for p in powers if p is not None]
                                        
                                        if len(temperatures) == 8 and len(powers) == 8:
                                            adjustments = controller.calculate_power_adjustments(
                                                shelf_id, temperatures, powers
                                            )
                                            
                                            temp_fields = [f"power_adj_{i+1}" for i in range(8)]
                                            insert_sql = text(f"""
                                                INSERT INTO control_commands (
                                                    device_id, shelf_id, auto_mode,
                                                    {', '.join(temp_fields)}
                                                ) VALUES (
                                                    :device_id, :shelf_id, :auto_mode,
                                                    {', '.join([f':{k}' for k in temp_fields])}
                                                )
                                            """)
                                            
                                            values = {
                                                "device_id": device_id,
                                                "shelf_id": shelf_id,
                                                "auto_mode": True,
                                                **{k: v for k, v in zip(temp_fields, adjustments)},
                                            }
                                            
                                            await db.execute(insert_sql, values)
                                            await db.commit()
                            except Exception as e:
                                print(f"Control loop error device {device_id} shelf {shelf_id}: {e}")
                                await db.rollback()
        except Exception as e:
            print(f"Background control loop error: {e}")
        
        await asyncio.sleep(settings.CONTROL_INTERVAL)


async def background_prediction_loop():
    while True:
        try:
            for device_id in range(1, 11):
                async with AsyncSessionLocal() as db:
                    try:
                        query = text(f"""
                            SELECT 
                                shelf_id, temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8,
                                vacuum_1, vacuum_2, cold_trap_temp,
                                power_1, power_2, power_3, power_4, power_5, power_6, power_7, power_8,
                                timestamp
                            FROM telemetry
                            WHERE device_id = :device_id AND timestamp >= :start_time
                            ORDER BY timestamp DESC
                            LIMIT 120
                        """)
                        
                        start_time = datetime.now() - timedelta(hours=2)
                        result = await db.execute(query, {
                            "device_id": device_id,
                            "start_time": start_time
                        })
                        rows = result.all()
                        
                        from app.api.prediction import prediction_service
                        
                        for row in rows:
                            temperatures = [row.temp_1, row.temp_2, row.temp_3, row.temp_4,
                                           row.temp_5, row.temp_6, row.temp_7, row.temp_8]
                            vacuum_levels = [row.vacuum_1, row.vacuum_2]
                            heating_powers = [row.power_1, row.power_2, row.power_3, row.power_4,
                                             row.power_5, row.power_6, row.power_7, row.power_8]
                            
                            temperatures = [t for t in temperatures if t is not None]
                            vacuum_levels = [v for v in vacuum_levels if v is not None]
                            heating_powers = [p for p in heating_powers if p is not None]
                            
                            if len(temperatures) == 8 and len(vacuum_levels) == 2 and len(heating_powers) == 8:
                                prediction_service.add_telemetry(
                                    device_id, row.shelf_id,
                                    temperatures, vacuum_levels,
                                    heating_powers, row.cold_trap_temp
                                )
                        
                        if len(rows) >= 10:
                            pred = prediction_service.predict(device_id)
                            
                            quality_alarm = alarm_detector.check_quality_prediction(
                                device_id, pred.is_qualified,
                                pred.moisture_content, pred.reconstitution_time
                            )
                            
                            if quality_alarm:
                                alarms = [quality_alarm]
                                await process_new_alarms(alarms, db)
                                
                                insert_sql = text("""
                                    INSERT INTO prediction_results (
                                        device_id, batch_id, timestamp,
                                        moisture_pred, moisture_conf, moisture_threshold,
                                        reconstitution_pred, reconstitution_conf, reconstitution_threshold,
                                        drying_rate, is_qualified
                                    ) VALUES (
                                        :device_id, :batch_id, :timestamp,
                                        :moisture_pred, :moisture_conf, :moisture_threshold,
                                        :reconstitution_pred, :reconstitution_conf, :reconstitution_threshold,
                                        :drying_rate, :is_qualified
                                    )
                                """)
                                
                                batch_id = f"BATCH-{device_id}-{datetime.now().strftime('%Y%m%d%H%M')}"
                                await db.execute(insert_sql, {
                                    "device_id": device_id,
                                    "batch_id": batch_id,
                                    "timestamp": datetime.now(),
                                    "moisture_pred": pred.moisture_content,
                                    "moisture_conf": pred.moisture_confidence,
                                    "moisture_threshold": pred.moisture_threshold,
                                    "reconstitution_pred": pred.reconstitution_time,
                                    "reconstitution_conf": pred.reconstitution_confidence,
                                    "reconstitution_threshold": pred.reconstitution_threshold,
                                    "drying_rate": pred.drying_rate,
                                    "is_qualified": pred.is_qualified
                                })
                                await db.commit()
                    except Exception as e:
                        print(f"Prediction loop error device {device_id}: {e}")
                        await db.rollback()
        except Exception as e:
            print(f"Background prediction loop error: {e}")
        
        await asyncio.sleep(settings.PREDICTION_INTERVAL)


async def background_alarm_check_loop():
    while True:
        try:
            for device_id in range(1, 11):
                async with AsyncSessionLocal() as db:
                    for shelf_id in range(1, 6):
                        try:
                            query = text(f"""
                                SELECT DISTINCT ON (shelf_id)
                                    temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8,
                                    vacuum_1, vacuum_2, cold_trap_temp
                                FROM telemetry
                                WHERE device_id = :device_id AND shelf_id = :shelf_id
                                  AND timestamp >= :start_time
                                ORDER BY shelf_id, timestamp DESC
                            """)
                            
                            start_time = datetime.now() - timedelta(minutes=2)
                            result = await db.execute(query, {
                                "device_id": device_id,
                                "shelf_id": shelf_id,
                                "start_time": start_time
                            })
                            row = result.first()
                            
                            if row:
                                temperatures = [row.temp_1, row.temp_2, row.temp_3, row.temp_4,
                                               row.temp_5, row.temp_6, row.temp_7, row.temp_8]
                                vacuum_levels = [row.vacuum_1, row.vacuum_2]
                                cold_trap_temp = row.cold_trap_temp
                                
                                temperatures = [t for t in temperatures if t is not None]
                                vacuum_levels = [v for v in vacuum_levels if v is not None]
                                
                                if len(temperatures) == 8 and len(vacuum_levels) == 2:
                                    alarms = alarm_detector.process_telemetry(
                                        device_id, shelf_id,
                                        temperatures, vacuum_levels, cold_trap_temp
                                    )
                                    
                                    if alarms:
                                        await process_new_alarms(alarms, db)
                        except Exception as e:
                            print(f"Alarm check error device {device_id} shelf {shelf_id}: {e}")
        except Exception as e:
            print(f"Background alarm loop error: {e}")
        
        await asyncio.sleep(10)


@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mqtt_connected": mqtt_service.is_available()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
