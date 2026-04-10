package com.example.fileprocessor.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * ════════════════════════════════════════════════════════════════
 *  THE BATON — passed between activities in the pipeline
 * ════════════════════════════════════════════════════════════════
 *
 * Think of this as a relay race baton:
 *   Activity 1 (SaveFile)   → fills: originalFileName, newFileName, localFilePath
 *   Activity 2 (OCR)        → fills: extractedText
 *   Activity 3 (Sentiment)  → fills: sentimentResult
 *
 * At the end, ALL fields are filled and returned to the REST controller.
 *
 * ⚠️ IMPORTANT TEMPORAL RULES for this class:
 *   ✅ Must have @NoArgsConstructor  → Temporal uses it for JSON deserialization
 *   ✅ Must have getters/setters     → @Data from Lombok provides these
 *   ✅ All fields must be serializable (String, int, List, etc. — no raw streams)
 *
 * HOW TO EXTEND:
 *   Adding a new activity (e.g. Translation)?
 *   Just add a new field here:  private String translatedText;
 *   Then fill it in your new activity implementation.
 */
@Data               // Lombok: generates getters, setters, equals, hashCode, toString
@NoArgsConstructor  // Required by Temporal for JSON deserialization
@AllArgsConstructor // Convenient for creating fully-populated objects in tests
public class FileProcessResult {

    // ── Filled by Activity 1: SaveFileActivity ───────────────────────────────
    private String originalFileName;
    // e.g. "invoice.pdf"

    private String newFileName;
    // e.g. "invoice_20240101123045789.pdf"  (original + unique timestamp suffix)

    private String localFilePath;
    // e.g. "/tmp/uploads/invoice_20240101123045789.pdf"

    // ── Filled by Activity 2: OcrActivity ────────────────────────────────────
    private String extractedText;
    // e.g. "Invoice #1234\nTotal due: $1,200\nDue date: Feb 1, 2024"

    // ── Filled by Activity 3: SentimentActivity ──────────────────────────────
    private String sentimentResult;
    // e.g. "NEUTRAL (confidence: 0.88)"

    // ── ADD NEW FIELDS HERE for future activities ─────────────────────────────
    // private String translatedText;       // for TranslationActivity
    // private String summaryText;          // for SummarizationActivity
    // private String category;             // for ClassificationActivity
}
