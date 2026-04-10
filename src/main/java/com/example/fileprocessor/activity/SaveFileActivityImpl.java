package com.example.fileprocessor.activity;

import com.example.fileprocessor.model.FileProcessResult;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * ════════════════════════════════════════════════════════════════
 *  ACTIVITY 1 IMPLEMENTATION — Save File Locally
 * ════════════════════════════════════════════════════════════════
 *
 * @Component → Spring manages this as a bean so TemporalConfig can inject it.
 *
 * WHAT THIS DOES STEP BY STEP:
 *   A. Creates /tmp/uploads/ directory if it doesn't exist
 *   B. Generates a unique filename by appending a timestamp suffix
 *      "invoice.pdf" → "invoice_20240101123045789.pdf"
 *   C. Writes fileContent to disk at the new path
 *   D. Returns FileProcessResult with localFilePath + newFileName filled in
 *
 * ERROR HANDLING:
 *   If any step throws an exception → Temporal automatically retries this
 *   activity up to maxAttempts (configured in FileWorkflowImpl).
 *   Each retry attempt appears as a separate event in Temporal UI.
 */
@Slf4j
@Component
public class SaveFileActivityImpl implements SaveFileActivity {

    @Value("${app.upload.dir:/tmp/uploads/}")
    private String uploadDir;

    @Override
    public FileProcessResult saveFile(String originalFileName, String fileContent) {
        log.info("▶ Activity 1 START: Saving file '{}'", originalFileName);

        try {
            // ── Step A: Ensure upload directory exists ───────────────────────
            Path uploadPath = Paths.get(uploadDir);
            if (!Files.exists(uploadPath)) {
                Files.createDirectories(uploadPath);
                log.debug("Created upload directory: {}", uploadDir);
            }

            // ── Step B: Generate unique filename with timestamp suffix ────────
            // Format: yyyyMMddHHmmssSSS  →  e.g. "20240101123045789"
            // Year(4) + Month(2) + Day(2) + Hour(2) + Min(2) + Sec(2) + Millis(3)
            // This gives millisecond-level uniqueness — safe for normal traffic.
            // For extremely high concurrency, consider UUID instead.
            String uniqueSuffix = LocalDateTime.now()
                    .format(DateTimeFormatter.ofPattern("yyyyMMddHHmmssSSS"));

            // Split "invoice.pdf" → name="invoice", ext=".pdf"
            String nameWithoutExt;
            String extension;
            if (originalFileName.contains(".")) {
                int dotIndex = originalFileName.lastIndexOf('.');
                nameWithoutExt = originalFileName.substring(0, dotIndex);
                extension = originalFileName.substring(dotIndex); // includes the dot
            } else {
                nameWithoutExt = originalFileName;
                extension = "";
            }

            // Final name: "invoice_20240101123045789.pdf"
            String newFileName = nameWithoutExt + "_" + uniqueSuffix + extension;

            // ── Step C: Write file content to disk ───────────────────────────
            String fullPath = uploadDir + newFileName;
            try (FileWriter writer = new FileWriter(fullPath)) {
                writer.write(fileContent);
            }

            log.info("✅ Activity 1 DONE: Saved as '{}' → '{}'", newFileName, fullPath);

            // ── Step D: Return the baton with file info filled in ────────────
            // extractedText and sentimentResult are null — OCR & Sentiment fill those
            return new FileProcessResult(
                    originalFileName,   // what the user uploaded
                    newFileName,        // our unique saved filename
                    fullPath,           // absolute path on disk
                    null,               // OCR will fill this in Activity 2
                    null                // Sentiment will fill this in Activity 3
            );

        } catch (Exception e) {
            log.error("❌ Activity 1 FAILED: {}", e.getMessage(), e);
            // Throwing RuntimeException signals Temporal to retry this activity
            throw new RuntimeException("SaveFile activity failed: " + e.getMessage(), e);
        }
    }
}
