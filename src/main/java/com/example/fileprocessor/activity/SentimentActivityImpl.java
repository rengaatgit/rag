package com.example.fileprocessor.activity;

import com.example.fileprocessor.model.FileProcessResult;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.Map;

/**
 * ════════════════════════════════════════════════════════════════
 *  ACTIVITY 3 IMPLEMENTATION — Sentiment Analysis
 * ════════════════════════════════════════════════════════════════
 *
 * WHAT THIS DOES:
 *   1. Picks up the baton (FileProcessResult) from Activity 2
 *   2. Sends extractedText to a Sentiment Analysis REST API
 *   3. Receives sentiment label + confidence score
 *   4. Formats and stores result in result.sentimentResult
 *   5. Returns the FULLY COMPLETE baton (all fields now populated)
 *
 * SENTIMENT API CONTRACT (expected by this impl):
 *   POST http://localhost:8082/sentiment/analyze
 *   Request body (JSON):
 *     { "text": "Invoice #1234\nTotal due: $1,200..." }
 *   Response body (JSON):
 *     { "sentiment": "NEUTRAL", "confidence": 0.88 }
 *
 * REPLACE THE API URL in application.properties:
 *   app.sentiment.api.url=http://your-sentiment-service/analyze
 *
 * ALTERNATIVE SENTIMENT PROVIDERS:
 *   AWS Comprehend:       https://docs.aws.amazon.com/comprehend
 *   Google Natural Language: https://cloud.google.com/natural-language
 *   Azure Text Analytics: https://azure.microsoft.com/en-us/products/ai-services/text-analytics
 *   HuggingFace Inference API: https://huggingface.co/inference-api
 */
@Slf4j
@Component
public class SentimentActivityImpl implements SentimentActivity {

    @Value("${app.sentiment.api.url:http://localhost:8082/sentiment/analyze}")
    private String sentimentApiUrl;

    private final WebClient webClient = WebClient.create();

    /**
     * Inner class to deserialize the Sentiment API JSON response.
     * Maps to: { "sentiment": "POSITIVE", "confidence": 0.92 }
     *
     * @Data from Lombok generates getters/setters needed by Jackson (JSON parser).
     */
    @Data
    static class SentimentResponse {
        private String sentiment;   // "POSITIVE", "NEGATIVE", or "NEUTRAL"
        private double confidence;  // 0.0 to 1.0
    }

    @Override
    public FileProcessResult analyzeSentiment(FileProcessResult result) {
        log.info("▶ Activity 3 START: Sentiment analysis on extracted text ({} chars)",
                result.getExtractedText() != null ? result.getExtractedText().length() : 0);

        try {
            // ── Step A: Validate we have text to analyze ─────────────────────
            if (result.getExtractedText() == null || result.getExtractedText().isBlank()) {
                throw new RuntimeException("Cannot analyze sentiment: extractedText is empty");
            }

            // ── Step B: Build request payload ────────────────────────────────
            Map<String, String> requestBody = Map.of(
                    "text", result.getExtractedText()
            );

            // ── Step C: Call Sentiment API ───────────────────────────────────
            SentimentResponse sentimentResponse = webClient
                    .post()
                    .uri(sentimentApiUrl)
                    .header("Content-Type", "application/json")
                    .bodyValue(requestBody)
                    .retrieve()
                    .bodyToMono(SentimentResponse.class)
                    .block(); // Synchronous — safe in Temporal Activity

            if (sentimentResponse == null) {
                throw new RuntimeException("Sentiment API returned null response");
            }

            // ── Step D: Format and store result ──────────────────────────────
            // e.g. "NEUTRAL (confidence: 0.88)"
            String sentimentResult = String.format("%s (confidence: %.2f)",
                    sentimentResponse.getSentiment().toUpperCase(),
                    sentimentResponse.getConfidence());

            log.info("✅ Activity 3 DONE: Sentiment = {}", sentimentResult);

            // ── Step E: Fill final field — baton is now 100% complete ─────────
            result.setSentimentResult(sentimentResult);
            return result;

        } catch (Exception e) {
            log.error("❌ Activity 3 FAILED: {}", e.getMessage(), e);
            throw new RuntimeException("Sentiment activity failed: " + e.getMessage(), e);
        }
    }
}
