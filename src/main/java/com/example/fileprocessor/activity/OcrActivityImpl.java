package com.example.fileprocessor.activity;

import com.example.fileprocessor.model.FileProcessResult;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.Map;

/**
 * ════════════════════════════════════════════════════════════════
 *  ACTIVITY 2 IMPLEMENTATION — OCR Text Extraction
 * ════════════════════════════════════════════════════════════════
 *
 * WHAT THIS DOES:
 *   1. Picks up the baton (FileProcessResult) from Activity 1
 *   2. Sends localFilePath + newFileName to an OCR REST API
 *   3. Receives extracted text from the API
 *   4. Stores the text in result.extractedText
 *   5. Returns updated result for Activity 3 to consume
 *
 * OCR API CONTRACT (expected by this impl):
 *   POST http://localhost:8081/ocr/extract
 *   Request body (JSON):
 *     { "filePath": "/tmp/uploads/invoice_xxx.pdf", "fileName": "invoice_xxx.pdf" }
 *   Response body (plain text):
 *     "Invoice #1234\nTotal due: $1,200\nDue date: Feb 1, 2024"
 *
 * REPLACE THE API URL in application.properties:
 *   app.ocr.api.url=http://your-ocr-service/extract
 *
 * ALTERNATIVE OCR PROVIDERS:
 *   Google Cloud Vision: https://cloud.google.com/vision/docs/ocr
 *   AWS Textract:        https://docs.aws.amazon.com/textract
 *   Azure Form Recognizer: https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence
 *   Tesseract REST:      https://github.com/hertzsprung/tesseract-rest
 */
@Slf4j
@Component
public class OcrActivityImpl implements OcrActivity {

    @Value("${app.ocr.api.url:http://localhost:8081/ocr/extract}")
    private String ocrApiUrl;

    // WebClient: Spring's non-blocking HTTP client
    // We use .block() to make it synchronous — safe inside Temporal Activity threads
    private final WebClient webClient = WebClient.create();

    @Override
    public FileProcessResult extractText(FileProcessResult result) {
        log.info("▶ Activity 2 START: OCR on file '{}'", result.getNewFileName());

        try {
            // ── Step A: Build the JSON request payload ───────────────────────
            // Tell OCR API where the file is
            Map<String, String> requestBody = Map.of(
                    "filePath", result.getLocalFilePath(),
                    "fileName", result.getNewFileName()
            );

            // ── Step B: Call OCR API ─────────────────────────────────────────
            // POST request → wait for response (synchronous via .block())
            String extractedText = webClient
                    .post()
                    .uri(ocrApiUrl)
                    .header("Content-Type", "application/json")
                    .bodyValue(requestBody)
                    .retrieve()
                    // If API returns 4xx/5xx, this throws WebClientResponseException
                    // which Temporal catches and retries automatically
                    .bodyToMono(String.class)
                    .block(); // Wait synchronously — safe in Temporal Activity

            if (extractedText == null || extractedText.isBlank()) {
                throw new RuntimeException("OCR API returned empty text for file: "
                        + result.getNewFileName());
            }

            log.info("✅ Activity 2 DONE: Extracted {} characters of text",
                    extractedText.length());

            // ── Step C: Fill extractedText into the baton ────────────────────
            result.setExtractedText(extractedText);
            return result;

        } catch (Exception e) {
            log.error("❌ Activity 2 FAILED: {}", e.getMessage(), e);
            throw new RuntimeException("OCR activity failed: " + e.getMessage(), e);
        }
    }
}
