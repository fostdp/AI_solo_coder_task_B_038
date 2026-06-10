from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import Optional, List
from datetime import datetime, timedelta
from pydantic import BaseModel
from app.core.database import get_db
import hashlib
import base64
import os


class DefectReviewRequest(BaseModel):
    reviewed: bool
    review_notes: Optional[str] = None


router = APIRouter(prefix="/api/defect", tags=["defect"])


@router.post("/upload", status_code=201)
async def upload_defect_image(
    device_id: int = Form(...),
    batch_id: str = Form(...),
    shelf_id: int = Form(...),
    vial_position: str = Form(...),
    file: Optional[UploadFile] = File(None),
    base64_data: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    try:
        if not file and not base64_data:
            raise HTTPException(status_code=400, detail="Either file or base64_data must be provided")
        
        image_data = None
        filename = None
        file_hash = None
        
        if file:
            image_data = await file.read()
            filename = file.filename
        elif base64_data:
            if base64_data.startswith("data:image"):
                base64_data = base64_data.split(",")[1]
            image_data = base64.b64decode(base64_data)
            filename = f"defect_{device_id}_{batch_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        
        file_hash = hashlib.sha256(image_data).hexdigest()
        
        upload_dir = "uploads/defects"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f"{file_hash}_{filename}")
        
        with open(file_path, "wb") as f:
            f.write(image_data)
        
        insert_upload_sql = text("""
            INSERT INTO image_uploads (
                device_id, batch_id, shelf_id, vial_position,
                filename, file_path, file_hash, file_size,
                upload_time, upload_source
            ) VALUES (
                :device_id, :batch_id, :shelf_id, :vial_position,
                :filename, :file_path, :file_hash, :file_size,
                :upload_time, :upload_source
            ) RETURNING id
        """)
        
        result = await db.execute(insert_upload_sql, {
            "device_id": device_id,
            "batch_id": batch_id,
            "shelf_id": shelf_id,
            "vial_position": vial_position,
            "filename": filename,
            "file_path": file_path,
            "file_hash": file_hash,
            "file_size": len(image_data),
            "upload_time": datetime.now(),
            "upload_source": "api"
        })
        
        upload_id = result.scalar()
        
        mock_defect_type = None
        mock_severity = None
        mock_confidence = None
        
        if hash(file_hash) % 3 == 0:
            mock_defect_type = "crack"
            mock_severity = "high"
            mock_confidence = 0.92
        elif hash(file_hash) % 3 == 1:
            mock_defect_type = "contamination"
            mock_severity = "medium"
            mock_confidence = 0.78
        
        insert_defect_sql = text("""
            INSERT INTO product_defects (
                upload_id, device_id, batch_id, shelf_id, vial_position,
                defect_type, severity, confidence, detected_time,
                is_reviewed, reviewed_by, reviewed_at, review_notes
            ) VALUES (
                :upload_id, :device_id, :batch_id, :shelf_id, :vial_position,
                :defect_type, :severity, :confidence, :detected_time,
                :is_reviewed, :reviewed_by, :reviewed_at, :review_notes
            ) RETURNING id
        """)
        
        defect_result = await db.execute(insert_defect_sql, {
            "upload_id": upload_id,
            "device_id": device_id,
            "batch_id": batch_id,
            "shelf_id": shelf_id,
            "vial_position": vial_position,
            "defect_type": mock_defect_type,
            "severity": mock_severity,
            "confidence": mock_confidence,
            "detected_time": datetime.now(),
            "is_reviewed": False,
            "reviewed_by": None,
            "reviewed_at": None,
            "review_notes": None
        })
        
        defect_id = defect_result.scalar()
        await db.commit()
        
        return {
            "status": "success",
            "upload_id": upload_id,
            "defect_id": defect_id,
            "defect_detected": mock_defect_type is not None,
            "defect_type": mock_defect_type,
            "severity": mock_severity,
            "confidence": mock_confidence,
            "file_hash": file_hash
        }
    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/detection/{device_id}")
async def get_detection_history(
    device_id: int,
    batch_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    try:
        conditions = ["pd.device_id = :device_id"]
        params = {"device_id": device_id, "limit": limit}
        
        if batch_id:
            conditions.append("pd.batch_id = :batch_id")
            params["batch_id"] = batch_id
        
        query = text(f"""
            SELECT 
                pd.id, pd.upload_id, pd.device_id, pd.batch_id,
                pd.shelf_id, pd.vial_position, pd.defect_type,
                pd.severity, pd.confidence, pd.detected_time,
                pd.is_reviewed, pd.reviewed_by, pd.reviewed_at,
                pd.review_notes, iu.filename, iu.file_path, iu.file_size
            FROM product_defects pd
            LEFT JOIN image_uploads iu ON pd.upload_id = iu.id
            WHERE {' AND '.join(conditions)}
            ORDER BY pd.detected_time DESC
            LIMIT :limit
        """)
        
        result = await db.execute(query, params)
        rows = result.all()
        
        defects = []
        for row in rows:
            defects.append({
                "id": row.id,
                "upload_id": row.upload_id,
                "device_id": row.device_id,
                "batch_id": row.batch_id,
                "shelf_id": row.shelf_id,
                "vial_position": row.vial_position,
                "defect_type": row.defect_type,
                "severity": row.severity,
                "confidence": row.confidence,
                "detected_time": row.detected_time,
                "is_reviewed": row.is_reviewed,
                "reviewed_by": row.reviewed_by,
                "reviewed_at": row.reviewed_at,
                "review_notes": row.review_notes,
                "filename": row.filename,
                "file_path": row.file_path,
                "file_size": row.file_size
            })
        
        return {"count": len(defects), "defects": defects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch/{batch_id}")
async def get_batch_defects(
    batch_id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        query = text(f"""
            SELECT 
                pd.id, pd.device_id, pd.shelf_id, pd.vial_position,
                pd.defect_type, pd.severity, pd.confidence, pd.detected_time,
                pd.is_reviewed, iu.filename
            FROM product_defects pd
            LEFT JOIN image_uploads iu ON pd.upload_id = iu.id
            WHERE pd.batch_id = :batch_id
            ORDER BY pd.detected_time DESC
        """)
        
        result = await db.execute(query, {"batch_id": batch_id})
        rows = result.all()
        
        defects = []
        defect_types = {}
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        reviewed_count = 0
        
        for row in rows:
            defects.append({
                "id": row.id,
                "device_id": row.device_id,
                "shelf_id": row.shelf_id,
                "vial_position": row.vial_position,
                "defect_type": row.defect_type,
                "severity": row.severity,
                "confidence": row.confidence,
                "detected_time": row.detected_time,
                "is_reviewed": row.is_reviewed,
                "filename": row.filename
            })
            
            if row.defect_type:
                defect_types[row.defect_type] = defect_types.get(row.defect_type, 0) + 1
            
            if row.severity:
                severity_counts[row.severity] = severity_counts.get(row.severity, 0) + 1
            
            if row.is_reviewed:
                reviewed_count += 1
        
        stats = {
            "total_defects": len(defects),
            "defect_types": defect_types,
            "severity_distribution": severity_counts,
            "reviewed_count": reviewed_count,
            "unreviewed_count": len(defects) - reviewed_count,
            "review_rate": round(reviewed_count / len(defects), 4) if defects else 0
        }
        
        return {"batch_id": batch_id, "statistics": stats, "defects": defects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/review/{defect_id}")
async def review_defect(
    defect_id: int,
    request: DefectReviewRequest,
    db: AsyncSession = Depends(get_db)
):
    try:
        check_query = text("""
            SELECT id FROM product_defects WHERE id = :defect_id
        """)
        
        result = await db.execute(check_query, {"defect_id": defect_id})
        row = result.first()
        
        if not row:
            raise HTTPException(status_code=404, detail="Defect not found")
        
        update_query = text("""
            UPDATE product_defects
            SET is_reviewed = :is_reviewed,
                review_notes = :review_notes,
                reviewed_at = :reviewed_at,
                reviewed_by = 'system'
            WHERE id = :defect_id
        """)
        
        await db.execute(update_query, {
            "defect_id": defect_id,
            "is_reviewed": request.reviewed,
            "review_notes": request.review_notes,
            "reviewed_at": datetime.now()
        })
        
        await db.commit()
        
        return {
            "status": "success",
            "defect_id": defect_id,
            "is_reviewed": request.reviewed,
            "reviewed_at": datetime.now()
        }
    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_defect_stats(
    hours: int = Query(24, ge=1, le=720),
    db: AsyncSession = Depends(get_db)
):
    try:
        start_time = datetime.now() - timedelta(hours=hours)
        
        query = text(f"""
            SELECT 
                COUNT(*) as total_uploads,
                COUNT(CASE WHEN defect_type IS NOT NULL THEN 1 END) as total_defects,
                COUNT(CASE WHEN is_reviewed = TRUE THEN 1 END) as reviewed_count,
                COUNT(CASE WHEN severity = 'high' THEN 1 END) as high_severity,
                COUNT(CASE WHEN severity = 'medium' THEN 1 END) as medium_severity,
                COUNT(CASE WHEN severity = 'low' THEN 1 END) as low_severity,
                defect_type,
                COUNT(*) as type_count,
                device_id,
                batch_id
            FROM product_defects
            WHERE detected_time >= :start_time
            GROUP BY ROLLUP (defect_type, device_id, batch_id)
            ORDER BY total_defects DESC
        """)
        
        result = await db.execute(query, {"start_time": start_time})
        rows = result.all()
        
        overall = {
            "time_window_hours": hours,
            "total_uploads": 0,
            "total_defects": 0,
            "defect_rate": 0,
            "reviewed_count": 0,
            "review_rate": 0,
            "severity_distribution": {
                "high": 0,
                "medium": 0,
                "low": 0
            }
        }
        
        by_type = {}
        by_device = {}
        by_batch = {}
        
        for row in rows:
            if row.defect_type is None and row.device_id is None and row.batch_id is None:
                overall["total_uploads"] = row.total_uploads or 0
                overall["total_defects"] = row.total_defects or 0
                overall["reviewed_count"] = row.reviewed_count or 0
                overall["severity_distribution"]["high"] = row.high_severity or 0
                overall["severity_distribution"]["medium"] = row.medium_severity or 0
                overall["severity_distribution"]["low"] = row.low_severity or 0
                
                if overall["total_uploads"] > 0:
                    overall["defect_rate"] = round(overall["total_defects"] / overall["total_uploads"], 4)
                    overall["review_rate"] = round(overall["reviewed_count"] / overall["total_defects"], 4) if overall["total_defects"] > 0 else 0
            
            elif row.defect_type and row.device_id is None and row.batch_id is None:
                by_type[row.defect_type] = row.type_count or 0
            
            elif row.device_id and row.defect_type is None and row.batch_id is None:
                by_device[str(row.device_id)] = row.total_defects or 0
            
            elif row.batch_id and row.defect_type is None and row.device_id is None:
                by_batch[row.batch_id] = row.total_defects or 0
        
        return {
            "overall": overall,
            "by_defect_type": by_type,
            "by_device": by_device,
            "by_batch": by_batch
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch_records")
async def get_batch_records(
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db)
):
    try:
        query = text(f"""
            SELECT 
                br.id, br.batch_id, br.product_name, br.manufacture_date,
                br.expiry_date, br.total_vials, br.device_id,
                COUNT(pd.id) as defect_count,
                COUNT(CASE WHEN pd.severity = 'high' THEN 1 END) as high_severity_count,
                COUNT(CASE WHEN pd.is_reviewed = TRUE THEN 1 END) as reviewed_count
            FROM batch_records br
            LEFT JOIN product_defects pd ON br.batch_id = pd.batch_id
            GROUP BY br.id, br.batch_id, br.product_name, br.manufacture_date,
                     br.expiry_date, br.total_vials, br.device_id
            ORDER BY br.manufacture_date DESC
            LIMIT :limit
        """)
        
        result = await db.execute(query, {"limit": limit})
        rows = result.all()
        
        records = []
        for row in rows:
            defect_count = row.defect_count or 0
            total_vials = row.total_vials or 0
            defect_rate = round(defect_count / total_vials, 4) if total_vials > 0 else 0
            review_rate = round((row.reviewed_count or 0) / defect_count, 4) if defect_count > 0 else 0
            
            records.append({
                "id": row.id,
                "batch_id": row.batch_id,
                "product_name": row.product_name,
                "manufacture_date": row.manufacture_date,
                "expiry_date": row.expiry_date,
                "total_vials": total_vials,
                "device_id": row.device_id,
                "defect_count": defect_count,
                "high_severity_count": row.high_severity_count or 0,
                "defect_rate": defect_rate,
                "reviewed_count": row.reviewed_count or 0,
                "review_rate": review_rate
            })
        
        return {"count": len(records), "records": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
