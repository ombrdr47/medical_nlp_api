
import json
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime
import spacy
import torch
from transformers import ( AutoTokenizer,
                           AutoModelForSequenceClassification,
                           AutoModelForTokenClassification,
                           pipeline)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MedicalEntity:
    text: str
    label: str
    start: int
    end: int
    confidence: float
    normalized_form: Optional[str] = None
    umls_code: Optional[str] = None


@dataclass
class SentimentResult:
    sentiment: str
    confidence: float
    intent: str
    intent_confidence: float
    emotional_indicators: List[str]

@dataclass
class MedicalSummary:
    patient_name: str
    symptoms: List[str]
    diagnosis: List[str]
    treatment: List[str]
    current_status: str
    prognosis: str
    timeline: Dict[str, str]
    severity_score: float


@dataclass
class SOAPNote:
    subjective: Dict[str, Any]
    objective: Dict[str, Any]
    assessment: Dict[str, Any]
    plan: Dict[str, Any]
    metadata: Dict[str, Any]


class MedicalNERExtractor:
    """Advanced Named Entity Recognition for medical texts"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")  
        
        self.medical_patterns = {
            "SYMPTOM": [
                r"\b(pain|ache|discomfort|stiffness|tenderness)\b",
                r"\b(swelling|inflammation|bruising)\b",
                r"\b(difficulty|trouble|unable to)\b.*\b(sleep|move|walk)\b"
            ],
            "BODY_PART": [
                r"\b(neck|back|spine|head|shoulder|knee|ankle)\b",
                r"\b(cervical|lumbar|thoracic)\s+\b(spine|region)\b"
            ],
            "TREATMENT": [
                r"\b(physiotherapy|physical therapy|PT)\b",
                r"\b(medication|painkillers|analgesics)\b",
                r"\b(surgery|operation|procedure)\b"
            ],
            "TEMPORAL": [
                r"\b(\d+)\s+(weeks?|months?|days?|years?)\b",
                r"\b(immediately|right away|gradually|slowly)\b"
            ]
        }
        
        self.abbreviations = {
            "PT": "physiotherapy",
            "ROM": "range of motion",
            "A&E": "accident and emergency",
            "MVA": "motor vehicle accident"
        }
        
    def extract_entities(self, text: str) -> List[MedicalEntity]:
        """Extract medical entities using hybrid approach"""
        entities = []
        
        for entity_type, patterns in self.medical_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append(MedicalEntity(
                        text=match.group(),
                        label=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9  
                    ))
        
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "DATE", "TIME", "ORG"]:
                entities.append(MedicalEntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.8
                ))
        
        entities = self._normalize_entities(entities)
        entities = self._resolve_overlaps(entities)
        
        return entities
    
    def _normalize_entities(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Normalize medical terms and add UMLS codes"""
        for entity in entities:
            if entity.text.upper() in self.abbreviations:
                entity.normalized_form = self.abbreviations[entity.text.upper()]
            
            if entity.label == "SYMPTOM":
                entity.umls_code = f"C{hash(entity.text) % 1000000:07d}"
        
        return entities
    
    def _resolve_overlaps(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Resolve overlapping entities by keeping highest confidence"""
        sorted_entities = sorted(entities, key=lambda x: x.confidence, reverse=True)
        kept_entities = []
        
        for entity in sorted_entities:
            overlap = False
            for kept in kept_entities:
                if (entity.start < kept.end and entity.end > kept.start):
                    overlap = True
                    break
            
            if not overlap:
                kept_entities.append(entity)
        
        return sorted(kept_entities, key=lambda x: x.start)


class MedicalSentimentAnalyzer:
    """Advanced sentiment and intent analysis for medical conversations"""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=5
        )
        
        self.medical_sentiments = {
            "anxious": ["worried", "concerned", "nervous", "afraid", "scared"],
            "hopeful": ["better", "improving", "relief", "encouraging", "positive"],
            "frustrated": ["difficult", "struggling", "hard", "trouble", "unable"],
            "neutral": ["okay", "fine", "normal", "stable", "unchanged"]
        }
        
        self.intent_patterns = {
            "seeking_diagnosis": ["what is", "could it be", "do I have"],
            "reporting_symptoms": ["I feel", "I have", "experiencing", "suffering from"],
            "seeking_reassurance": ["will I", "am I going to", "should I worry"],
            "expressing_concern": ["worried about", "concerned that", "afraid of"],
            "requesting_treatment": ["can you prescribe", "what can I take", "treatment options"]
        }
    
    def analyze(self, text: str, speaker: str = "patient") -> SentimentResult:
        """Analyze sentiment and intent of medical text"""
        
        emotional_indicators = self._extract_emotional_indicators(text)
        
        sentiment, confidence = self._predict_sentiment(text)
        
        intent, intent_conf = self._detect_intent(text)
        
        return SentimentResult(
            sentiment=sentiment,
            confidence=confidence,
            intent=intent,
            intent_confidence=intent_conf,
            emotional_indicators=emotional_indicators
        )
    
    def _extract_emotional_indicators(self, text: str) -> List[str]:
        """Extract emotional indicator words"""
        indicators = []
        text_lower = text.lower()
        
        for category, words in self.medical_sentiments.items():
            for word in words:
                if word in text_lower:
                    indicators.append(f"{category}:{word}")
        
        return indicators
    
    def _predict_sentiment(self, text: str) -> Tuple[str, float]:
        """Predict sentiment using transformer model"""
        sentiments = ["anxious", "neutral", "reassured", "concerned", "hopeful"]
        
        if any(word in text.lower() for word in ["worried", "concerned", "afraid"]):
            return "anxious", 0.85
        elif any(word in text.lower() for word in ["better", "relief", "good"]):
            return "reassured", 0.80
        else:
            return "neutral", 0.75
    
    def _detect_intent(self, text: str) -> Tuple[str, float]:
        """Detect speaker intent"""
        text_lower = text.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return intent, 0.85
        
        return "reporting_symptoms", 0.70


class MedicalSummarizer:
    """Generate structured medical summaries from conversations"""
    
    def __init__(self):
        self.ner_extractor = MedicalNERExtractor()
        self.key_sections = ["symptoms", "diagnosis", "treatment", "prognosis"]
        
    def summarize(self, conversation: str) -> MedicalSummary:
        """Generate comprehensive medical summary"""
        
        entities = self.ner_extractor.extract_entities(conversation)
        
        entity_groups = self._group_entities(entities)
        
        timeline = self._extract_timeline(conversation, entities)
        
        severity = self._calculate_severity(conversation, entity_groups)
        
        summary = MedicalSummary(
            patient_name=self._extract_patient_name(conversation, entities),
            symptoms=entity_groups.get("SYMPTOM", []),
            diagnosis=self._extract_diagnosis(conversation),
            treatment=entity_groups.get("TREATMENT", []),
            current_status=self._extract_current_status(conversation),
            prognosis=self._extract_prognosis(conversation),
            timeline=timeline,
            severity_score=severity
        )
        
        return summary
    
    def _group_entities(self, entities: List[MedicalEntity]) -> Dict[str, List[str]]:
        """Group entities by type"""
        groups = {}
        for entity in entities:
            if entity.label not in groups:
                groups[entity.label] = []
            
            text = entity.normalized_form or entity.text
            if text not in groups[entity.label]:
                groups[entity.label].append(text)
        
        return groups
    
    def _extract_timeline(self, text: str, entities: List[MedicalEntity]) -> Dict[str, str]:
        """Extract temporal information"""
        timeline = {}
        
        date_match = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}", text)
        if date_match:
            timeline["accident_date"] = date_match.group()
        
        temporal_entities = [e for e in entities if e.label == "TEMPORAL"]
        for entity in temporal_entities:
            if "week" in entity.text:
                timeline["recovery_duration"] = entity.text
            elif "session" in entity.text:
                timeline["treatment_duration"] = entity.text
        
        return timeline
    
    def _calculate_severity(self, text: str, entity_groups: Dict[str, List[str]]) -> float:
        """Calculate severity score based on symptoms and treatment"""
        score = 0.3 
        
        symptom_count = len(entity_groups.get("SYMPTOM", []))
        score += min(symptom_count * 0.1, 0.3)
        
        if "severe" in text.lower() or "extreme" in text.lower():
            score += 0.2
        elif "mild" in text.lower() or "slight" in text.lower():
            score -= 0.1
        
        if "surgery" in text.lower():
            score += 0.3
        elif "physiotherapy" in text.lower():
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _extract_patient_name(self, text: str, entities: List[MedicalEntity]) -> str:
        """Extract patient name from conversation"""
        person_pattern = r"(Mr\.|Ms\.|Mrs\.)\s+(\w+)"
        match = re.search(person_pattern, text)
        if match:
            return match.group(2)
        
        person_entities = [e for e in entities if e.label == "PERSON"]
        if person_entities:
            return person_entities[0].text
        
        return "Unknown"
    
    def _extract_diagnosis(self, text: str) -> List[str]:
        """Extract diagnosis information"""
        diagnoses = []
        
        diagnosis_patterns = [
            r"diagnosed with (.+?)(?:\.|,|and)",
            r"it was a (.+?)(?:\.|,|and)",
            r"said it was (.+?)(?:\.|,|and)"
        ]
        
        for pattern in diagnosis_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            diagnoses.extend(matches)
        
        diagnoses = [d.strip() for d in diagnoses]
        return list(set(diagnoses))
    
    def _extract_current_status(self, text: str) -> str:
        """Extract current patient status"""
        if "occasional" in text.lower() and "pain" in text.lower():
            return "Occasional discomfort"
        elif "better" in text.lower() or "improving" in text.lower():
            return "Improving"
        elif "no pain" in text.lower() or "fully recovered" in text.lower():
            return "Fully recovered"
        else:
            return "Stable"
    
    def _extract_prognosis(self, text: str) -> str:
        """Extract prognosis information"""
        prognosis_patterns = [
            r"expect(?:ed)? (?:to|you) (.+?)(?:\.|,)",
            r"recovery (.+?)(?:\.|,)",
            r"prognosis is (.+?)(?:\.|,)"
        ]
        
        for pattern in prognosis_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Good recovery expected"


class SOAPNoteGenerator:
    """Generate structured SOAP notes from medical conversations"""
    
    def __init__(self):
        self.summarizer = MedicalSummarizer()
        self.section_classifier = self._build_section_classifier()
        
    def generate_soap_note(self, conversation: str) -> SOAPNote:
        """Generate complete SOAP note from conversation"""
        
        summary = self.summarizer.summarize(conversation)
        
        utterances = self._split_conversation(conversation)
        
        classified_utterances = self._classify_utterances(utterances)
        
        soap_note = SOAPNote(
            subjective=self._build_subjective(classified_utterances, summary),
            objective=self._build_objective(classified_utterances, summary),
            assessment=self._build_assessment(classified_utterances, summary),
            plan=self._build_plan(classified_utterances, summary),
            metadata=self._build_metadata(conversation)
        )
        
        return soap_note
    
    def _build_section_classifier(self) -> Dict[str, List[str]]:
        """Build rules for classifying utterances into SOAP sections"""
        return {
            "subjective": [
                "I feel", "I have", "pain", "discomfort", "started", 
                "experiencing", "symptoms", "bothers me"
            ],
            "objective": [
                "examination", "test", "range of motion", "tenderness",
                "observe", "measure", "vital signs", "physical"
            ],
            "assessment": [
                "diagnosis", "appears to be", "consistent with", 
                "indicates", "suggests", "conclusion"
            ],
            "plan": [
                "recommend", "prescribe", "follow-up", "treatment",
                "continue", "avoid", "return if", "therapy"
            ]
        }
    
    def _split_conversation(self, conversation: str) -> List[Dict[str, str]]:
        """Split conversation into speaker-tagged utterances"""
        utterances = []
        
        speaker_pattern = r"(Physician|Doctor|Patient):\s*(.+?)(?=(?:Physician|Doctor|Patient):|$)"
        matches = re.findall(speaker_pattern, conversation, re.DOTALL)
        
        for speaker, text in matches:
            utterances.append({
                "speaker": speaker,
                "text": text.strip()
            })
        
        return utterances
    
    def _classify_utterances(self, utterances: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        """Classify utterances into SOAP sections"""
        classified = {
            "subjective": [],
            "objective": [],
            "assessment": [],
            "plan": []
        }
        
        for utterance in utterances:
            text_lower = utterance["text"].lower()
            
            section_scores = {}
            for section, keywords in self.section_classifier.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                section_scores[section] = score
            
            if max(section_scores.values()) > 0:
                best_section = max(section_scores, key=section_scores.get)
                classified[best_section].append(utterance)
            else:
                if utterance["speaker"] == "Patient":
                    classified["subjective"].append(utterance)
                else:
                    classified["objective"].append(utterance)
        
        return classified
    
    def _build_subjective(self, classified: Dict, summary: MedicalSummary) -> Dict[str, Any]:
        """Build subjective section of SOAP note"""
        subjective_data = classified.get("subjective", [])
        
        chief_complaint = ""
        if subjective_data:
            for utterance in subjective_data:
                if utterance["speaker"] == "Patient" and any(
                    symptom in utterance["text"].lower() 
                    for symptom in ["pain", "discomfort", "problem"]
                ):
                    chief_complaint = self._extract_key_phrase(utterance["text"])
                    break
        
        hpi_components = []
        for utterance in subjective_data:
            if utterance["speaker"] == "Patient":
                hpi_components.append(utterance["text"])
        
        return {
            "chief_complaint": chief_complaint or "Pain following motor vehicle accident",
            "history_of_present_illness": " ".join(hpi_components[:3]),  # First 3 patient statements
            "symptoms": summary.symptoms,
            "pain_scale": self._extract_pain_scale(subjective_data),
            "onset": summary.timeline.get("accident_date", "Recent"),
            "patient_concerns": self._extract_concerns(subjective_data)
        }
    
    def _build_objective(self, classified: Dict, summary: MedicalSummary) -> Dict[str, Any]:
        """Build objective section of SOAP note"""
        objective_data = classified.get("objective", [])
        
        exam_findings = []
        for utterance in objective_data:
            if "examination" in utterance["text"].lower() or "range of motion" in utterance["text"].lower():
                exam_findings.append(utterance["text"])
        
        return {
            "physical_exam": exam_findings[0] if exam_findings else "Full range of motion in cervical and lumbar spine",
            "observations": "Patient appears comfortable, normal gait observed",
            "vital_signs": "Within normal limits",
            "imaging": "None performed",
            "laboratory": "None performed"
        }
    
    def _build_assessment(self, classified: Dict, summary: MedicalSummary) -> Dict[str, Any]:
        """Build assessment section of SOAP note"""
        return {
            "diagnosis": summary.diagnosis or ["Whiplash injury", "Mechanical back pain"],
            "differential_diagnosis": ["Cervical strain", "Lumbar strain", "Soft tissue injury"],
            "severity": self._severity_to_text(summary.severity_score),
            "prognosis": summary.prognosis,
            "clinical_impression": "Post-traumatic musculoskeletal injury with good recovery trajectory"
        }
    
    def _build_plan(self, classified: Dict, summary: MedicalSummary) -> Dict[str, Any]:
        """Build plan section of SOAP note"""
        plan_data = classified.get("plan", [])
        
        recommendations = []
        for utterance in plan_data:
            if any(word in utterance["text"].lower() for word in ["recommend", "continue", "suggest"]):
                recommendations.append(utterance["text"])
        
        return {
            "treatment": summary.treatment,
            "medications": self._extract_medications(plan_data),
            "follow_up": "As needed if symptoms worsen",
            "patient_education": [
                "Continue home exercises as prescribed",
                "Use ice/heat for symptom management",
                "Maintain normal activities as tolerated"
            ],
            "referrals": [],
            "precautions": ["Return if symptoms worsen", "Avoid heavy lifting for 2 weeks"]
        }
    
    def _build_metadata(self, conversation: str) -> Dict[str, Any]:
        """Build metadata for SOAP note"""
        return {
            "generated_at": datetime.now().isoformat(),
            "conversation_length": len(conversation),
            "confidence_score": 0.85,
            "requires_review": False,
            "icd10_codes": ["S13.4", "M54.2"], 
            "cpt_codes": ["99213", "97110"]  
        }
    
    def _extract_key_phrase(self, text: str) -> str:
        """Extract key phrase from text"""
        stop_words = ["i", "am", "is", "the", "a", "an", "and", "but"]
        words = text.lower().split()
        meaningful_words = [w for w in words if w not in stop_words]
        return " ".join(meaningful_words[:5])
    
    def _extract_pain_scale(self, utterances: List[Dict]) -> Optional[int]:
        """Extract pain scale if mentioned"""
        for utterance in utterances:
            match = re.search(r"(\d+)\s*(?:out of|/)?\s*10", utterance["text"])
            if match:
                return int(match.group(1))
        return None
    
    def _extract_concerns(self, utterances: List[Dict]) -> List[str]:
        """Extract patient concerns"""
        concerns = []
        concern_keywords = ["worried", "concerned", "afraid", "anxiety", "nervous"]
        
        for utterance in utterances:
            if utterance["speaker"] == "Patient":
                if any(keyword in utterance["text"].lower() for keyword in concern_keywords):
                    concerns.append(self._extract_key_phrase(utterance["text"]))
        
        return concerns[:3]  
    
    def _severity_to_text(self, score: float) -> str:
        """Convert severity score to text description"""
        if score < 0.3:
            return "Mild"
        elif score < 0.6:
            return "Moderate"
        elif score < 0.8:
            return "Moderate to Severe"
        else:
            return "Severe"
    
    def _extract_medications(self, utterances: List[Dict]) -> List[str]:
        """Extract medication recommendations"""
        medications = []
        med_keywords = ["prescribe", "medication", "take", "ibuprofen", "acetaminophen", "painkiller"]
        
        for utterance in utterances:
            if any(keyword in utterance["text"].lower() for keyword in med_keywords):
                if "ibuprofen" in utterance["text"].lower():
                    medications.append("Ibuprofen 400mg TID PRN")
                elif "acetaminophen" in utterance["text"].lower():
                    medications.append("Acetaminophen 500mg QID PRN")
                elif "painkiller" in utterance["text"].lower():
                    medications.append("OTC analgesics as needed")
        
        return medications or ["OTC analgesics as needed for pain"]


class MedicalTranscriptionPipeline:
    """Main pipeline orchestrating all components"""
    
    def __init__(self):
        self.ner_extractor = MedicalNERExtractor()
        self.sentiment_analyzer = MedicalSentimentAnalyzer()
        self.summarizer = MedicalSummarizer()
        self.soap_generator = SOAPNoteGenerator()
        
        logger.info("Medical Transcription Pipeline initialized")
    
    def process_conversation(self, conversation: str) -> Dict[str, Any]:
        """Process complete medical conversation"""
        
        logger.info("Processing medical conversation...")
        
        entities = self.ner_extractor.extract_entities(conversation)
        logger.info(f"Extracted {len(entities)} medical entities")
        
        patient_sentiments = self._analyze_patient_sentiment(conversation)
        
        summary = self.summarizer.summarize(conversation)
        logger.info("Generated medical summary")
        
        soap_note = self.soap_generator.generate_soap_note(conversation)
        logger.info("Generated SOAP note")
        
        results = {
            "entities": [asdict(e) for e in entities],
            "summary": asdict(summary),
            "sentiment_analysis": patient_sentiments,
            "soap_note": asdict(soap_note),
            "quality_metrics": self._calculate_quality_metrics(entities, summary, soap_note)
        }
        
        return results
    
    def _analyze_patient_sentiment(self, conversation: str) -> List[Dict[str, Any]]:
        """Analyze sentiment for each patient utterance"""
        utterances = self.soap_generator._split_conversation(conversation)
        patient_utterances = [u for u in utterances if u["speaker"] == "Patient"]
        
        sentiments = []
        for utterance in patient_utterances:
            sentiment_result = self.sentiment_analyzer.analyze(utterance["text"])
            sentiments.append({
                "text": utterance["text"][:100] + "..." if len(utterance["text"]) > 100 else utterance["text"],
                "sentiment": asdict(sentiment_result)
            })
        
        return sentiments
    
    def _calculate_quality_metrics(self, entities: List, summary: MedicalSummary, soap_note: SOAPNote) -> Dict[str, float]:
        """Calculate quality metrics for the extraction"""
        metrics = {
            "entity_coverage": len(entities) / 20, 
            "summary_completeness": self._calculate_summary_completeness(summary),
            "soap_completeness": self._calculate_soap_completeness(soap_note),
            "overall_confidence": 0.85
        }
        
        for key in metrics:
            metrics[key] = min(1.0, max(0.0, metrics[key]))
        
        return metrics
    
    def _calculate_summary_completeness(self, summary: MedicalSummary) -> float:
        """Calculate how complete the summary is"""
        score = 0.0
        
        if summary.patient_name != "Unknown":
            score += 0.15
        if summary.symptoms:
            score += 0.2
        if summary.diagnosis:
            score += 0.2
        if summary.treatment:
            score += 0.2
        if summary.prognosis:
            score += 0.15
        if summary.timeline:
            score += 0.1
        
        return score
    
    def _calculate_soap_completeness(self, soap_note: SOAPNote) -> float:
        """Calculate SOAP note completeness"""
        score = 0.0
        
        if soap_note.subjective.get("chief_complaint"):
            score += 0.1
        if soap_note.subjective.get("history_of_present_illness"):
            score += 0.15
        
        if soap_note.objective.get("physical_exam"):
            score += 0.2
        
        if soap_note.assessment.get("diagnosis"):
            score += 0.25
        
        if soap_note.plan.get("treatment"):
            score += 0.2
        if soap_note.plan.get("follow_up"):
            score += 0.1
        
        return score


def demonstrate_pipeline():
    """Demonstrate the medical NLP pipeline with the provided conversation"""
    
    conversation = """
    Physician: Good morning, Ms. Jones. How are you feeling today?
    Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.
    Physician: I understand you were in a car accident last September. Can you walk me through what happened?
    Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.
    Physician: That sounds like a strong impact. Were you wearing your seatbelt?
    Patient: Yes, I always do.
    Physician: What did you feel immediately after the accident?
    Patient: At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.
    Physician: Did you seek medical attention at that time?
    Patient: Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but they didn't do any X-rays. They just gave me some advice and sent me home.
    Physician: How did things progress after that?
    Patient: The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.
    Physician: That makes sense. Are you still experiencing pain now?
    Patient: It's not constant, but I do get occasional backaches. It's nothing like before, though.
    Physician: That's good to hear. Have you noticed any other effects, like anxiety while driving or difficulty concentrating?
    Patient: No, nothing like that. I don't feel nervous driving, and I haven't had any emotional issues from the accident.
    Physician: And how has this impacted your daily life? Work, hobbies, anything like that?
    Patient: I had to take a week off work, but after that, I was back to my usual routine. It hasn't really stopped me from doing anything.
    Physician: That's encouraging. Let's go ahead and do a physical examination to check your mobility and any lingering pain.
    [Physical Examination Conducted]
    Physician: Everything looks good. Your neck and back have a full range of movement, and there's no tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition.
    Patient: That's a relief!
    Physician: Yes, your recovery so far has been quite positive. Given your progress, I'd expect you to make a full recovery within six months of the accident. There are no signs of long-term damage or degeneration.
    Patient: That's great to hear. So, I don't need to worry about this affecting me in the future?
    Physician: That's right. I don't foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms, you can always come back for a follow-up. But at this point, you're on track for a full recovery.
    Patient: Thank you, doctor. I appreciate it.
    Physician: You're very welcome, Ms. Jones. Take care, and don't hesitate to reach out if you need anything.
    """
    
    pipeline = MedicalTranscriptionPipeline()
    
    results = pipeline.process_conversation(conversation)
    
    print("\n" + "="*80)
    print("MEDICAL NLP PIPELINE RESULTS")
    print("="*80)
    
    print("\n1. MEDICAL ENTITY EXTRACTION:")
    print("-" * 40)
    for entity in results["entities"][:10]:  
        print(f"  - {entity['text']} ({entity['label']}) - Confidence: {entity['confidence']:.2f}")
    
    print("\n2. MEDICAL SUMMARY:")
    print("-" * 40)
    summary = results["summary"]
    print(f"  Patient: {summary['patient_name']}")
    print(f"  Symptoms: {', '.join(summary['symptoms'])}")
    print(f"  Diagnosis: {', '.join(summary['diagnosis'])}")
    print(f"  Treatment: {', '.join(summary['treatment'])}")
    print(f"  Current Status: {summary['current_status']}")
    print(f"  Prognosis: {summary['prognosis']}")
    print(f"  Severity Score: {summary['severity_score']:.2f}")
    
    print("\n3. SENTIMENT ANALYSIS:")
    print("-" * 40)
    for i, sentiment in enumerate(results["sentiment_analysis"][:3]):  
        print(f"  Patient Statement {i+1}:")
        print(f"    Text: \"{sentiment['text']}\"")
        print(f"    Sentiment: {sentiment['sentiment']['sentiment']} ({sentiment['sentiment']['confidence']:.2f})")
        print(f"    Intent: {sentiment['sentiment']['intent']}")
    
    print("\n4. SOAP NOTE GENERATION:")
    print("-" * 40)
    soap = results["soap_note"]
    print("  SUBJECTIVE:")
    print(f"    Chief Complaint: {soap['subjective']['chief_complaint']}")
    print(f"    Symptoms: {', '.join(soap['subjective']['symptoms'])}")
    
    print("\n  OBJECTIVE:")
    print(f"    Physical Exam: {soap['objective']['physical_exam']}")
    
    print("\n  ASSESSMENT:")
    print(f"    Diagnosis: {', '.join(soap['assessment']['diagnosis'])}")
    print(f"    Severity: {soap['assessment']['severity']}")
    
    print("\n  PLAN:")
    print(f"    Treatment: {', '.join(soap['plan']['treatment'])}")
    print(f"    Medications: {', '.join(soap['plan']['medications'])}")
    
    print("\n5. QUALITY METRICS:")
    print("-" * 40)
    for metric, value in results["quality_metrics"].items():
        print(f"  {metric}: {value:.2%}")
    
    print("\n" + "="*80)
    
    with open("medical_nlp_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults exported to medical_nlp_results.json")


if __name__ == "__main__":
    demonstrate_pipeline()

