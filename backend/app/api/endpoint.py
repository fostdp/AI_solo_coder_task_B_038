from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List, Optional
from datetime import datetime, timedelta
from app.core.database import get_db
from app.core.config import settings
import json
import uuid

try:
    import redis.asyncio as redis_async
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

router = APIRouter(prefix="/api/endpoint", tags=["endpoint"])


class RedisPublisher:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._client = None
        return cls._instance

    async def _get_client(self):
        if not HAS_REDIS:
            return None
        if self._client is None:
            try:
                self._client = redis_async.Redis(
                    host="localhost",
                    port=6379,
                    db=0,
                    decode_responses=True
                )
                await self._client.ping()
            except Exception:
                self._client = None
        return self._client

    async def publish(self, channel: str, message: dict) -> bool:
        client = await self._get_client()
        if not client:
            return False
        try:
            await client.publish(channel, json.dumps(message, ensure_ascii=False))
            return True
        except Exception:
            self._client = None
            return False


redis_publisher = RedisPublisher()


@router.get("/detection/{device_id}")
async def get_detection_history(
    device_id: int,
    batch_id: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    try:
        if not start_time:
            start_time = datetime.now() - timedelta(days=7)
        if not end_time:
            end_time = datetime.now()

        conditions = [
            "device_id = :device_id",
            "endpoint_timestamp >= :start_time",
            "endpoint_timestamp <= :end_time"
        ]
        params = {
            "device_id": device_id,
            "start_time": start_time,
            "end_time": end_time
        }

        if batch_id:
            conditions.append("batch_id = :batch_id")
            params["batch_id"] = batch_id

        where_clause = " AND ".join(conditions)

        query = text(f"""
            SELECT
                id, device_id, batch_id, cycle_phase, detection_method,
                endpoint_timestamp, detection_confidence, pressure_rise_delta,
                temp_inflection_point, temp_first_derivative, autoencoder_recon_error,
                cycle_duration_hours, estimated_energy_saving, is_accepted, created_at
            FROM drying_endpoints
            WHERE {where_clause}
            ORDER BY endpoint_timestamp DESC
            LIMIT :limit
        """)
        params["limit"] = limit

        result = await db.execute(query, params)
        rows = result.all()

        data = []
        for row in rows:
            data.append({
                "id": str(row.id),
                "device_id": row.device_id,
                "batch_id": row.batch_id,
                "cycle_phase": row.cycle_phase,
                "detection_method": row.detection_method,
                "endpoint_timestamp": row.endpoint_timestamp.isoformat() if row.endpoint_timestamp else None,
                "detection_confidence": row.detection_confidence,
                "pressure_rise_delta": row.pressure_rise_delta,
                "temp_inflection_point": row.temp_inflection_point,
                "temp_first_derivative": row.temp_first_derivative,
                "autoencoder_recon_error": row.autoencoder_recon_error,
                "cycle_duration_hours": row.cycle_duration_hours,
                "estimated_energy_saving": row.estimated_energy_saving,
                "is_accepted": row.is_accepted,
                "created_at": row.created_at.isoformat() if row.created_at else None
            })

        return {"count": len(data), "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current/{device_id}")
async def get_current_status(
    device_id: int,
    db: AsyncSession = Depends(get_db)
):
    try:
        query = text(f"""
            SELECT
                br.batch_id, br.batch_status,
                br.primary_drying_endpoint, br.secondary_drying_endpoint,
                br.start_time, br.total_cycle_hours,
                de.cycle_phase as last_detected_phase,
                de.endpoint_timestamp as last_endpoint_time,
                de.detection_confidence as last_confidence,
                de.detection_method as last_method
            FROM batch_records br
            LEFT JOIN LATERAL (
                SELECT cycle_phase, endpoint_timestamp, detection_confidence, detection_method
                FROM drying_endpoints
                WHERE device_id = :device_id
                  AND batch_id = br.batch_id
                ORDER BY endpoint_timestamp DESC
                LIMIT 1
            ) de ON true
            WHERE br.device_id = :device_id
            ORDER BY br.start_time DESC
            LIMIT 1
        """)

        result = await db.execute(query, {"device_id": device_id})
        row = result.first()

        if not row:
            return {
                "device_id": device_id,
                "current_phase": "idle",
                "batch_status": "idle",
                "batch_id": None,
                "primary_endpoint_detected": False,
                "secondary_endpoint_detected": False,
                "last_endpoint_detection": None
            }

        primary_detected = row.primary_drying_endpoint is not None
        secondary_detected = row.secondary_drying_endpoint is not None

        if row.batch_status == "running":
            if secondary_detected:
                current_phase = "secondary_drying"
            elif primary_detected:
                current_phase = "secondary_drying"
            else:
                current_phase = "primary_drying"
        else:
            current_phase = row.batch_status or "idle"

        last_detection = None
        if row.last_endpoint_time:
            last_detection = {
                "phase": row.last_detected_phase,
                "timestamp": row.last_endpoint_time.isoformat() if row.last_endpoint_time else None,
                "confidence": row.last_confidence,
                "method": row.last_method
            }

        return {
            "device_id": device_id,
            "batch_id": row.batch_id,
            "batch_status": row.batch_status,
            "current_phase": current_phase,
            "primary_endpoint_detected": primary_detected,
            "secondary_endpoint_detected": secondary_detected,
            "primary_drying_endpoint": row.primary_drying_endpoint.isoformat() if row.primary_drying_endpoint else None,
            "secondary_drying_endpoint": row.secondary_drying_endpoint.isoformat() if row.secondary_drying_endpoint else None,
            "cycle_start_time": row.start_time.isoformat() if row.start_time else None,
            "total_cycle_hours": row.total_cycle_hours,
            "last_endpoint_detection": last_detection
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prt/{device_id}", status_code=202)
async def trigger_prt(
    device_id: int,
    db: AsyncSession = Depends(get_db)
):
    try:
        check_query = text(f"""
            SELECT batch_id, batch_status, primary_drying_endpoint, secondary_drying_endpoint
            FROM batch_records
            WHERE device_id = :device_id
            ORDER BY start_time DESC
            LIMIT 1
        """)
        result = await db.execute(check_query, {"device_id": device_id})
        row = result.first()

        if not row or row.batch_status not in ["running"]:
            raise HTTPException(
                status_code=400,
                detail="Device is not in an active drying phase"
            )

        batch_id = row.batch_id
        if row.secondary_drying_endpoint:
            current_phase = "completed"
        elif row.primary_drying_endpoint:
            current_phase = "secondary_drying"
        else:
            current_phase = "primary_drying"

        if current_phase not in ["primary_drying", "secondary_drying"]:
            raise HTTPException(
                status_code=400,
                detail="Device is not in an active drying phase"
            )

        insert_query = text(f"""
            INSERT INTO pressure_rise_tests (
                device_id, batch_id, test_start_time
            ) VALUES (
                :device_id, :batch_id, :start_time
            )
            RETURNING id
        """)

        start_time = datetime.now()
        result = await db.execute(insert_query, {
            "device_id": device_id,
            "batch_id": batch_id,
            "start_time": start_time
        })
        test_id = result.scalar()
        await db.commit()

        message = {
            "header": {
                "message_id": str(uuid.uuid4()),
                "message_type": "pressure_rise_test",
                "source_service": "api-gateway",
                "target_service": None,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0"
            },
            "payload": {
                "command": "start",
                "device_id": device_id,
                "batch_id": batch_id,
                "test_id": str(test_id),
                "current_phase": current_phase,
                "requested_at": start_time.isoformat()
            }
        }

        published = await redis_publisher.publish("endpoint:prt", message)

        return {
            "status": "accepted",
            "test_id": str(test_id),
            "device_id": device_id,
            "batch_id": batch_id,
            "current_phase": current_phase,
            "requested_at": start_time.isoformat(),
            "message_published": published
        }
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prt/{device_id}")
async def get_prt_history(
    device_id: int,
    batch_id: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db)
):
    try:
        if not start_time:
            start_time = datetime.now() - timedelta(days=7)
        if not end_time:
            end_time = datetime.now()

        conditions = [
            "device_id = :device_id",
            "test_start_time >= :start_time",
            "test_start_time <= :end_time"
        ]
        params = {
            "device_id": device_id,
            "start_time": start_time,
            "end_time": end_time
        }

        if batch_id:
            conditions.append("batch_id = :batch_id")
            params["batch_id"] = batch_id

        where_clause = " AND ".join(conditions)

        query = text(f"""
            SELECT
                id, device_id, batch_id, test_start_time, test_end_time,
                initial_pressure_pa, final_pressure_pa, pressure_rise_pa_per_min,
                test_duration_seconds, is_endpoint_detected, detection_confidence,
                created_at
            FROM pressure_rise_tests
            WHERE {where_clause}
            ORDER BY test_start_time DESC
            LIMIT :limit
        """)
        params["limit"] = limit

        result = await db.execute(query, params)
        rows = result.all()

        data = []
        for row in rows:
            if row.test_end_time:
                test_status = "completed"
            else:
                test_status = "in_progress"
            data.append({
                "id": str(row.id),
                "device_id": row.device_id,
                "batch_id": row.batch_id,
                "test_start_time": row.test_start_time.isoformat() if row.test_start_time else None,
                "test_end_time": row.test_end_time.isoformat() if row.test_end_time else None,
                "initial_pressure_pa": row.initial_pressure_pa,
                "final_pressure_pa": row.final_pressure_pa,
                "pressure_rise_pa_per_min": row.pressure_rise_pa_per_min,
                "test_duration_seconds": row.test_duration_seconds,
                "is_endpoint_detected": row.is_endpoint_detected,
                "detection_confidence": row.detection_confidence,
                "test_status": test_status,
                "created_at": row.created_at.isoformat() if row.created_at else None
            })

        return {"count": len(data), "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{device_id}")
async def get_endpoint_stats(
    device_id: int,
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db)
):
    try:
        start_time = datetime.now() - timedelta(days=days)

        detection_query = text(f"""
            SELECT
                COUNT(*) as total_detections,
                COUNT(CASE WHEN is_accepted = true THEN 1 END) as accepted_detections,
                COUNT(CASE WHEN cycle_phase = 'primary_drying' THEN 1 END) as primary_detections,
                COUNT(CASE WHEN cycle_phase = 'secondary_drying' THEN 1 END) as secondary_detections,
                AVG(detection_confidence) as avg_confidence,
                AVG(cycle_duration_hours) as avg_cycle_duration,
                AVG(estimated_energy_saving) as avg_energy_saving,
                SUM(estimated_energy_saving) as total_energy_saving,
                AVG(temp_first_derivative) as avg_temp_derivative,
                AVG(autoencoder_recon_error) as avg_recon_error
            FROM drying_endpoints
            WHERE device_id = :device_id
              AND endpoint_timestamp >= :start_time
        """)

        result = await db.execute(detection_query, {
            "device_id": device_id,
            "start_time": start_time
        })
        det_row = result.first()

        prt_query = text(f"""
            SELECT
                COUNT(*) as total_prt_tests,
                COUNT(CASE WHEN is_endpoint_detected = true THEN 1 END) as endpoint_detected_tests,
                AVG(pressure_rise_pa_per_min) as avg_pressure_rise,
                AVG(test_duration_seconds) as avg_test_duration,
                AVG(detection_confidence) as avg_prt_confidence
            FROM pressure_rise_tests
            WHERE device_id = :device_id
              AND test_start_time >= :start_time
        """)

        result = await db.execute(prt_query, {
            "device_id": device_id,
            "start_time": start_time
        })
        prt_row = result.first()

        batch_query = text(f"""
            SELECT
                COUNT(*) as total_batches,
                COUNT(CASE WHEN batch_status = 'completed' THEN 1 END) as completed_batches,
                AVG(total_cycle_hours) as avg_total_cycle_hours,
                AVG(EXTRACT(EPOCH FROM (secondary_drying_endpoint - primary_drying_endpoint))/3600) 
                    FILTER (WHERE primary_drying_endpoint IS NOT NULL 
                           AND secondary_drying_endpoint IS NOT NULL) 
                    as avg_phase_transition_hours
            FROM batch_records
            WHERE device_id = :device_id
              AND start_time >= :start_time
        """)

        result = await db.execute(batch_query, {
            "device_id": device_id,
            "start_time": start_time
        })
        batch_row = result.first()

        method_query = text(f"""
            SELECT
                detection_method,
                COUNT(*) as count,
                AVG(detection_confidence) as avg_confidence
            FROM drying_endpoints
            WHERE device_id = :device_id
              AND endpoint_timestamp >= :start_time
            GROUP BY detection_method
            ORDER BY count DESC
        """)

        result = await db.execute(method_query, {
            "device_id": device_id,
            "start_time": start_time
        })
        method_rows = result.all()

        method_breakdown = []
        for row in method_rows:
            method_breakdown.append({
                "method": row.detection_method,
                "count": row.count,
                "avg_confidence": round(float(row.avg_confidence), 3) if row.avg_confidence else None
            })

        total_detections = det_row.total_detections or 0
        accepted_rate = (det_row.accepted_detections or 0) / total_detections * 100 if total_detections > 0 else 0

        default_primary_duration = 24.0
        default_secondary_duration = 8.0
        avg_primary_duration = None
        avg_secondary_duration = None
        cycle_time_reduction = None

        if det_row.avg_cycle_duration and det_row.primary_detections and det_row.secondary_detections:
            if det_row.primary_detections > 0:
                primary_dur_query = text(f"""
                    SELECT AVG(cycle_duration_hours) as avg_primary
                    FROM drying_endpoints
                    WHERE device_id = :device_id
                      AND cycle_phase = 'primary_drying'
                      AND endpoint_timestamp >= :start_time
                """)
                result = await db.execute(primary_dur_query, {
                    "device_id": device_id,
                    "start_time": start_time
                })
                pd = result.first()
                avg_primary_duration = pd.avg_primary if pd.avg_primary else None

            if det_row.secondary_detections > 0:
                secondary_dur_query = text(f"""
                    SELECT AVG(cycle_duration_hours) as avg_secondary
                    FROM drying_endpoints
                    WHERE device_id = :device_id
                      AND cycle_phase = 'secondary_drying'
                      AND endpoint_timestamp >= :start_time
                """)
                result = await db.execute(secondary_dur_query, {
                    "device_id": device_id,
                    "start_time": start_time
                })
                sd = result.first()
                avg_secondary_duration = sd.avg_secondary if sd.avg_secondary else None

            if avg_primary_duration and avg_secondary_duration:
                actual_total = avg_primary_duration + avg_secondary_duration
                default_total = default_primary_duration + default_secondary_duration
                cycle_time_reduction = ((default_total - actual_total) / default_total) * 100

        return {
            "device_id": device_id,
            "time_window_days": days,
            "detection_stats": {
                "total_detections": total_detections,
                "accepted_detections": det_row.accepted_detections or 0,
                "acceptance_rate": round(accepted_rate, 2),
                "primary_drying_detections": det_row.primary_detections or 0,
                "secondary_drying_detections": det_row.secondary_detections or 0,
                "average_confidence": round(float(det_row.avg_confidence), 3) if det_row.avg_confidence else None,
                "average_cycle_duration_hours": round(float(det_row.avg_cycle_duration), 2) if det_row.avg_cycle_duration else None,
                "average_primary_duration_hours": round(float(avg_primary_duration), 2) if avg_primary_duration else None,
                "average_secondary_duration_hours": round(float(avg_secondary_duration), 2) if avg_secondary_duration else None,
                "method_breakdown": method_breakdown
            },
            "prt_stats": {
                "total_tests": prt_row.total_prt_tests or 0,
                "endpoint_detected_tests": prt_row.endpoint_detected_tests or 0,
                "average_pressure_rise_pa_per_min": round(float(prt_row.avg_pressure_rise), 4) if prt_row.avg_pressure_rise else None,
                "average_test_duration_seconds": round(float(prt_row.avg_test_duration), 1) if prt_row.avg_test_duration else None,
                "average_confidence": round(float(prt_row.avg_prt_confidence), 3) if prt_row.avg_prt_confidence else None
            },
            "batch_stats": {
                "total_batches": batch_row.total_batches or 0,
                "completed_batches": batch_row.completed_batches or 0,
                "average_total_cycle_hours": round(float(batch_row.avg_total_cycle_hours), 2) if batch_row.avg_total_cycle_hours else None,
                "average_phase_transition_hours": round(float(batch_row.avg_phase_transition_hours), 2) if batch_row.avg_phase_transition_hours else None
            },
            "efficiency_stats": {
                "cycle_time_reduction_percent": round(float(cycle_time_reduction), 2) if cycle_time_reduction else None,
                "average_energy_saving_kwh": round(float(det_row.avg_energy_saving), 2) if det_row.avg_energy_saving else None,
                "total_energy_saving_kwh": round(float(det_row.total_energy_saving), 2) if det_row.total_energy_saving else None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
