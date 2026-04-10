# File Processor — Spring Boot + Temporal

A Spring Boot application that processes uploaded files through a 3-step Temporal workflow:
1. **Save File** — saves locally with a unique timestamp suffix
2. **OCR** — extracts text via OCR API
3. **Sentiment Analysis** — analyzes sentiment of extracted text

---

## Project Structure

```
src/main/java/com/example/fileprocessor/
│
├── config/
│   └── TemporalConfig.java          ← ONE-TIME: Temporal beans (Client, Worker)
│
├── workflow/
│   ├── FileWorkflow.java            ← ONE-TIME: Workflow interface
│   └── FileWorkflowImpl.java        ← ONE-TIME: Orchestrator (chains activities)
│
├── activity/
│   ├── SaveFileActivity.java        ← REPEATABLE: Activity interface
│   ├── SaveFileActivityImpl.java    ← REPEATABLE: Activity logic
│   ├── OcrActivity.java             ← REPEATABLE: Activity interface
│   ├── OcrActivityImpl.java         ← REPEATABLE: Activity logic
│   ├── SentimentActivity.java       ← REPEATABLE: Activity interface
│   └── SentimentActivityImpl.java   ← REPEATABLE: Activity logic
│
├── model/
│   └── FileProcessResult.java       ← Shared result baton between activities
│
├── controller/
│   └── FileController.java          ← REST PUT /api/files/process
│
└── FileProcessorApplication.java    ← Spring Boot entry point
```

---

## Prerequisites

- Java 17+
- Maven 3.8+
- Docker (for Temporal Server)

---

## Quick Start

### 1. Start Temporal Server (Docker)

```bash
docker run --rm -p 7233:7233 -p 8080:8080 temporalio/auto-setup:latest
```

Temporal UI available at: http://localhost:8080

### 2. Build the Project

```bash
mvn clean install
```

### 3. Run the Application

```bash
mvn spring-boot:run
```

### 4. Test the Endpoint

```bash
curl -X PUT "http://localhost:8080/api/files/process" \
     -d "fileName=invoice.pdf" \
     -d "fileContent=Invoice+%231234%0ATotal+due%3A+%241200"
```

**Expected Response:**
```json
{
  "originalFileName": "invoice.pdf",
  "newFileName": "invoice_20240101123045789.pdf",
  "localFilePath": "/tmp/uploads/invoice_20240101123045789.pdf",
  "extractedText": "Invoice #1234\nTotal due: $1,200",
  "sentimentResult": "NEUTRAL (confidence: 0.88)"
}
```

---

## Configuration (application.properties)

| Property | Default | Description |
|---|---|---|
| `spring.temporal.connection.target` | `127.0.0.1:7233` | Temporal server address |
| `spring.temporal.namespace` | `default` | Temporal namespace |
| `app.ocr.api.url` | `http://localhost:8081/ocr/extract` | OCR API endpoint |
| `app.sentiment.api.url` | `http://localhost:8082/sentiment/analyze` | Sentiment API endpoint |
| `app.upload.dir` | `/tmp/uploads/` | Local file storage directory |

---

## Adding a New Activity (3 Steps)

Example: Adding a **Translation** activity.

### Step 1 — Create Interface (2 min)

```java
// TranslationActivity.java
@ActivityInterface
public interface TranslationActivity {
    @ActivityMethod
    FileProcessResult translate(FileProcessResult result);
}
```

### Step 2 — Create Implementation (your logic)

```java
// TranslationActivityImpl.java
@Slf4j
@Component
public class TranslationActivityImpl implements TranslationActivity {
    @Override
    public FileProcessResult translate(FileProcessResult result) {
        // Call your translation API here
        // result.setTranslatedText("...");
        return result;
    }
}
```

Also add `private String translatedText;` to `FileProcessResult.java`.

### Step 3 — Wire Up (2 lines total)

In `TemporalConfig.java`:
```java
// Add parameter:
TranslationActivityImpl translationActivity

// Add to registerActivitiesImplementations():
worker.registerActivitiesImplementations(
    saveFileActivity, ocrActivity, sentimentActivity,
    translationActivity   // ← ADD HERE
);
```

In `FileWorkflowImpl.java`:
```java
result = translationActivity.translate(result);   // ← ADD ONE LINE
```

---

## Temporal UI

After starting a workflow, view real-time execution at:
```
http://localhost:8080/namespaces/default/workflows
```

You'll see:
- **Workflow Run** with unique ID
- **Event History**: ActivityTaskScheduled → Started → Completed (for each activity)
- **Retry events** if any activity fails
- **Worker** status and heartbeat

---

## OCR & Sentiment API Contract

### OCR API
```
POST /ocr/extract
Content-Type: application/json

Request:  { "filePath": "/tmp/uploads/invoice_xxx.pdf", "fileName": "invoice_xxx.pdf" }
Response: "Extracted text content here..."
```

### Sentiment API
```
POST /sentiment/analyze
Content-Type: application/json

Request:  { "text": "Extracted text content here..." }
Response: { "sentiment": "POSITIVE", "confidence": 0.92 }
```

---

## Retry Policy

All activities are configured with:
- **Max attempts**: 3
- **Initial interval**: 2 seconds
- **Backoff**: 2x (2s → 4s → 8s)
- **Timeout per attempt**: 60 seconds

Configure in `FileWorkflowImpl.java` → `DEFAULT_OPTIONS`.
