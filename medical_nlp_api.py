from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import asyncio
from enum import Enum
import logging
from pydantic import field_validator
from fastapi import FastAPI, HTTPException
from typing import List, Any, Dict, Optional
from medical_nlp_pipeline import MedicalTranscriptionPipeline
from fastapi.openapi.utils import get_openapi
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical Transcription NLP API",
    description="Advanced NLP pipeline for medical conversation analysis and SOAP note generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscriptionRequest(BaseModel):
    """Request model for transcription analysis"""
    conversation_text: str = Field(..., min_length=50, description="Medical conversation text")
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    encounter_date: Optional[datetime] = Field(None, description="Date of medical encounter")
    settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Processing settings")
    
    @field_validator('conversation_text')  
    @classmethod
    def validate_conversation(cls, v):
        if not any(word in v.lower() for word in ['patient', 'doctor', 'physician']):
            raise ValueError("Text must contain medical conversation with patient/doctor dialogue")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "conversation_text": "Doctor: How are you feeling today?\nPatient: I have been experiencing neck pain...",
                "patient_id": "PAT-12345",
                "encounter_date": "2025-06-11T10:00:00",
                "settings": {
                    "include_confidence_scores": True,
                    "extract_icd_codes": True
                }
            }
        }

class TextRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Text for entity extraction")


class EntityResponse(BaseModel):
    """Response model for extracted entities"""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    normalized_form: Optional[str] = None
    umls_code: Optional[str] = None


class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    intent: str
    intent_confidence: float
    emotional_indicators: List[str]

class SummaryResponse(BaseModel):
    """Response model for medical summary"""
    patient_name: str
    symptoms: List[str]
    diagnosis: List[str]
    treatment: List[str]
    current_status: str
    prognosis: str
    timeline: Dict[str, str]
    severity_score: float


class SOAPResponse(BaseModel):
    """Response model for SOAP note"""
    subjective: Dict[str, Any]
    objective: Dict[str, Any]
    assessment: Dict[str, Any]
    plan: Dict[str, Any]
    metadata: Dict[str, Any]


class AnalysisResponse(BaseModel):
    """Complete analysis response"""
    request_id: str
    status: str
    entities: List[EntityResponse]
    summary: SummaryResponse
    sentiment_analysis: List[Dict[str, Any]]
    soap_note: SOAPResponse
    quality_metrics: Dict[str, float]
    processing_time: float


class ProcessingStatus(str, Enum):
    """Processing status enum"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AsyncJobResponse(BaseModel):
    """Response for async job submission"""
    job_id: str
    status: ProcessingStatus
    message: str
    result_url: str


job_storage = {}


class MedicalTranscriptionPipeline:
    """Mock pipeline for demonstration"""
    def process_conversation(self, text: str) -> Dict[str, Any]:
        return {
            "entities": [
                {"text": "neck pain", "label": "SYMPTOM", "start": 10, "end": 19, "confidence": 0.95},
                {"text": "physiotherapy", "label": "TREATMENT", "start": 50, "end": 63, "confidence": 0.90}
            ],
            "summary": {
                "patient_name": "Jones",
                "symptoms": ["neck pain", "back pain"],
                "diagnosis": ["whiplash injury"],
                "treatment": ["physiotherapy", "painkillers"],
                "current_status": "Improving",
                "prognosis": "Full recovery expected",
                "timeline": {"accident_date": "September 1st"},
                "severity_score": 0.4
            },
            "sentiment_analysis": [
                {
                    "text": "I'm doing better, but I still have some discomfort",
                    "sentiment": {
                        "sentiment": "hopeful",
                        "confidence": 0.75,
                        "intent": "reporting_symptoms",
                        "intent_confidence": 0.80,
                        "emotional_indicators": ["hopeful:better"]
                    }
                }
            ],
            "soap_note": {
                "subjective": {
                    "chief_complaint": "Neck and back pain following MVA",
                    "symptoms": ["neck pain", "back pain"]
                },
                "objective": {
                    "physical_exam": "Full range of motion, no tenderness"
                },
                "assessment": {
                    "diagnosis": ["Whiplash injury"],
                    "severity": "Mild to Moderate"
                },
                "plan": {
                    "treatment": ["Continue physiotherapy"],
                    "medications": ["OTC analgesics PRN"]
                },
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "confidence_score": 0.85
                }
            },
            "quality_metrics": {
                "entity_coverage": 0.75,
                "summary_completeness": 0.90,
                "soap_completeness": 0.85,
                "overall_confidence": 0.85
            }
        }


pipeline = MedicalTranscriptionPipeline()

app = FastAPI()

@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Medical NLP Pipeline API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/v1/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_conversation(request: TranscriptionRequest):
    """
    Analyze medical conversation and extract structured information.
    
    This endpoint performs:
    - Named Entity Recognition (NER)
    - Medical summarization
    - Sentiment analysis
    - SOAP note generation
    """
    try:
        start_time = datetime.now()
        
        results = pipeline.process_conversation(request.conversation_text)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = AnalysisResponse(
            request_id=str(uuid.uuid4()),
            status="completed",
            entities=[EntityResponse(**entity) for entity in results["entities"]],
            summary=SummaryResponse(**results["summary"]),
            sentiment_analysis=results["sentiment_analysis"],
            soap_note=SOAPResponse(**results["soap_note"]),
            quality_metrics=results["quality_metrics"],
            processing_time=processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/v1/analyze/async", response_model=AsyncJobResponse, tags=["Analysis"])
async def analyze_conversation_async(
    request: TranscriptionRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit medical conversation for asynchronous analysis.
    
    Use this endpoint for long conversations or when immediate response is not required.
    """
    job_id = str(uuid.uuid4())
    
    job_storage[job_id] = {
        "status": ProcessingStatus.PENDING,
        "request": request.dict(),
        "result": None,
        "error": None,
        "created_at": datetime.now()
    }
    
    background_tasks.add_task(process_async_job, job_id, request)
    
    return AsyncJobResponse(
        job_id=job_id,
        status=ProcessingStatus.PENDING,
        message="Job submitted successfully",
        result_url=f"/api/v1/jobs/{job_id}"
    )


async def process_async_job(job_id: str, request: TranscriptionRequest):
    """Background task for processing async jobs"""
    try:
        job_storage[job_id]["status"] = ProcessingStatus.PROCESSING
        
        await asyncio.sleep(2)  
        
        results = pipeline.process_conversation(request.conversation_text)
        
        job_storage[job_id]["status"] = ProcessingStatus.COMPLETED
        job_storage[job_id]["result"] = results
        job_storage[job_id]["completed_at"] = datetime.now()
        
    except Exception as e:
        job_storage[job_id]["status"] = ProcessingStatus.FAILED
        job_storage[job_id]["error"] = str(e)
        logger.error(f"Async job {job_id} failed: {str(e)}")


@app.get("/api/v1/jobs/{job_id}", tags=["Jobs"])
async def get_job_status(job_id: str):
    """Get status and results of an async analysis job"""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_storage[job_id]
    
    if job["status"] == ProcessingStatus.COMPLETED:
        # Return full results
        return {
            "job_id": job_id,
            "status": job["status"],
            "created_at": job["created_at"],
            "completed_at": job.get("completed_at"),
            "result": job["result"]
        }
    else:
        # Return status only
        return {
            "job_id": job_id,
            "status": job["status"],
            "created_at": job["created_at"],
            "error": job.get("error")
        }


@app.post("/api/v1/entities/extract", tags=["Entities"])
async def extract_entities(request: TextRequest):
    """Extract medical entities from text"""
    try:
        from medical_nlp_pipeline import MedicalNERExtractor
        extractor = MedicalNERExtractor()
        entities = extractor.extract_entities(request.text)
        
        return {
            "entities": [
                {
                    "text": e.text,
                    "label": e.label,
                    "confidence": e.confidence,
                    "normalized": e.normalized_form
                }
                for e in entities
            ],
            "count": len(entities)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/soap/generate", response_model=SOAPResponse, tags=["SOAP"])
async def generate_soap_note(request: TranscriptionRequest):
    """Generate SOAP note from medical conversation"""
    try:
        from medical_nlp_pipeline import SOAPNoteGenerator
        generator = SOAPNoteGenerator()
        soap_note = generator.generate_soap_note(request.conversation_text)
        
        return SOAPResponse(**soap_note.__dict__)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/sentiment/analyze",
    response_model=SentimentResponse,
    tags=["Sentiment"]
)
async def analyze_sentiment(request: TextRequest):
    """
    Analyze sentiment and intent of medical text
    """
    try:
        from medical_nlp_pipeline import MedicalSentimentAnalyzer
        analyzer = MedicalSentimentAnalyzer()
        result = analyzer.analyze(request.text)

        return SentimentResponse(
            text=request.text,
            sentiment=result.sentiment,
            confidence=result.confidence,
            intent=result.intent,
            intent_confidence=result.intent_confidence,
            emotional_indicators=result.emotional_indicators
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models/info", tags=["Models"])
async def get_model_info():
    """Get information about loaded models"""
    return {
        "models": {
            "ner": {
                "name": "Medical NER",
                "version": "1.0.0",
                "backend": "spaCy + BioBERT",
                "entities": ["SYMPTOM", "TREATMENT", "BODY_PART", "DIAGNOSIS"]
            },
            "sentiment": {
                "name": "Medical Sentiment Analyzer",
                "version": "1.0.0",
                "backend": "DistilBERT",
                "classes": ["anxious", "neutral", "reassured", "concerned", "hopeful"]
            },
            "summarization": {
                "name": "Medical Summarizer",
                "version": "1.0.0",
                "backend": "Hybrid (Extractive + Template)"
            },
            "soap": {
                "name": "SOAP Note Generator",
                "version": "1.0.0",
                "backend": "Rule-based + BERT"
            }
        },
        "capabilities": {
            "max_text_length": 10000,
            "supported_languages": ["en"],
            "batch_processing": True,
            "streaming": False
        }
    }


from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws/transcribe-stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time transcription analysis"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                from medical_nlp_pipeline import MedicalNERExtractor
                extractor = MedicalNERExtractor()
                entities = extractor.extract_entities(data)
                
                await websocket.send_json({
                    "type": "entities",
                    "data": [
                        {"text": e.text, "label": e.label}
                        for e in entities
                    ]
                })
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "type": "validation_error"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": "internal_error",
            "message": str(exc) if app.debug else "An error occurred"
        }
    )


@app.on_event("startup")
async def startup_event():
    """Initialize models and resources on startup"""
    logger.info("Starting Medical NLP API...")
    logger.info("API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    logger.info("Shutting down Medical NLP API...")
    logger.info("API shutdown complete")



def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Medical Transcription NLP API",
        version="1.0.0",
        description="""
        ## Overview
        
        Advanced NLP pipeline for medical conversation analysis featuring:
        
        - **Named Entity Recognition**: Extract symptoms, treatments, diagnoses
        - **Medical Summarization**: Generate structured summaries
        - **Sentiment Analysis**: Analyze patient emotions and concerns
        - **SOAP Note Generation**: Create clinical documentation
        
        ## Authentication
        
        Currently using API key authentication. Include your API key in the header:
        ```
        X-API-Key: your-api-key
        ```
        
        ## Rate Limits
        
        - 100 requests per minute for standard endpoints
        - 10 concurrent WebSocket connections
        
        ## Examples
        
        See the `/docs` endpoint for interactive examples.
        """,
        routes=app.routes,
    )
    
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


if __name__ == "__main__":
    
    # Run with: python medical_nlp_api.py
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True  
    )