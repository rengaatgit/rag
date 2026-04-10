package com.example.fileprocessor.workflow;

import com.example.fileprocessor.activity.OcrActivity;
import com.example.fileprocessor.activity.SaveFileActivity;
import com.example.fileprocessor.activity.SentimentActivity;
import com.example.fileprocessor.model.FileProcessResult;
import io.temporal.activity.ActivityOptions;
import io.temporal.common.RetryOptions;
import io.temporal.workflow.Workflow;
import org.slf4j.Logger;

import java.time.Duration;

/**
 * ════════════════════════════════════════════════════════════════
 *  WORKFLOW IMPLEMENTATION — THE ORCHESTRATOR
 *  ONE-TIME SETUP — only add new activities here in future
 * ════════════════════════════════════════════════════════════════
 *
 * MENTAL MODEL:
 *   Think of this as a MANAGER who delegates tasks.
 *   The manager doesn't write files, call OCR, or call sentiment APIs.
 *   The manager says: "SaveFile team, do your job. OCR team, your turn. Sentiment team, finish it."
 *
 * ══ STRICT TEMPORAL WORKFLOW RULES ══════════════════════════════
 *   These rules exist because Temporal REPLAYS workflow code on worker restart.
 *   For replay to work correctly, workflow code must be deterministic.
 *
 *   ❌ NEVER: Thread.sleep()          → ✅ USE: Workflow.sleep()
 *   ❌ NEVER: new Date() / Instant.now() → ✅ USE: Workflow.currentTimeMillis()
 *   ❌ NEVER: Math.random() / UUID.randomUUID() → ✅ USE: Workflow.newRandom()
 *   ❌ NEVER: Direct HTTP calls (WebClient, RestTemplate, OkHttp)
 *             → ✅ PUT all I/O inside Activity implementations
 *   ❌ NEVER: System.out.println()    → ✅ USE: Workflow.getLogger()
 *   ✅ SAFE:  Calling activity stubs (Temporal manages these)
 *   ✅ SAFE:  Simple Java logic (if/else, loops, string ops)
 * ════════════════════════════════════════════════════════════════
 *
 * HOW TO ADD A NEW ACTIVITY (3 steps):
 *   Step 1: Add stub field below (copy any existing stub, 2 lines)
 *   Step 2: Call it in processFile() after sentimentActivity (1 line)
 *   Step 3: Register impl in TemporalConfig.java (1 line)
 */
public class FileWorkflowImpl implements FileWorkflow {

    // Use Temporal's replay-safe logger, NOT LoggerFactory.getLogger()
    private static final Logger log = Workflow.getLogger(FileWorkflowImpl.class);

    // ── ActivityOptions: Timeout + Retry configuration ──────────────────────
    //
    // startToCloseTimeout → Max time allowed for ONE attempt of the activity.
    //                        If OCR API hangs for >60s → Temporal cancels & retries.
    //
    // setMaximumAttempts(3) → Try up to 3 times before marking activity as FAILED.
    //
    // setBackoffCoefficient(2.0) → Exponential backoff between retries:
    //                              Retry 1: wait 2s
    //                              Retry 2: wait 4s  (2 * 2.0)
    //                              Retry 3: wait 8s  (4 * 2.0)
    //
    // All retry attempts are visible in Temporal UI as separate events.
    //
    // TIP: You can define different options per activity if some need longer timeouts:
    //   private static final ActivityOptions OCR_OPTIONS = ActivityOptions.newBuilder()
    //       .setStartToCloseTimeout(Duration.ofMinutes(5))  // OCR can be slow
    //       ...build();
    private static final ActivityOptions DEFAULT_OPTIONS = ActivityOptions.newBuilder()
            .setStartToCloseTimeout(Duration.ofSeconds(60))
            .setRetryOptions(
                RetryOptions.newBuilder()
                    .setMaximumAttempts(3)
                    .setInitialInterval(Duration.ofSeconds(2))
                    .setBackoffCoefficient(2.0)
                    .setMaximumInterval(Duration.ofSeconds(30))
                    .build()
            )
            .build();

    // ── Activity Stubs ───────────────────────────────────────────────────────
    //
    // A "stub" is Temporal's PROXY to an activity.
    // When you call stub.saveFile(...), Temporal:
    //   1. Serializes the arguments to JSON
    //   2. Sends a task to the Temporal Server (task queue)
    //   3. Worker picks up the task and runs SaveFileActivityImpl.saveFile()
    //   4. Serializes the return value to JSON
    //   5. Returns it here as a FileProcessResult object
    //
    // This means the activity could run on a DIFFERENT machine/pod
    // and Temporal guarantees the result comes back correctly.
    //
    // ── ADD NEW STUB HERE when adding a future activity ──────────────────────
    private final SaveFileActivity saveFileActivity =
            Workflow.newActivityStub(SaveFileActivity.class, DEFAULT_OPTIONS);

    private final OcrActivity ocrActivity =
            Workflow.newActivityStub(OcrActivity.class, DEFAULT_OPTIONS);

    private final SentimentActivity sentimentActivity =
            Workflow.newActivityStub(SentimentActivity.class, DEFAULT_OPTIONS);

    // Future activity stub (uncomment when you create TranslationActivity):
    // private final TranslationActivity translationActivity =
    //         Workflow.newActivityStub(TranslationActivity.class, DEFAULT_OPTIONS);


    @Override
    public FileProcessResult processFile(String originalFileName, String fileContent) {
        log.info("═══ Workflow START: Processing file '{}' ═══", originalFileName);

        // ── Activity 1: Save File ─────────────────────────────────────────────
        // Temporal event logged: ActivityTaskScheduled → ActivityTaskStarted → ActivityTaskCompleted
        // If it fails → Temporal retries up to 3 times automatically
        log.info("→ Step 1: Saving file locally...");
        FileProcessResult result = saveFileActivity.saveFile(originalFileName, fileContent);
        log.info("← Step 1 complete: saved as '{}'", result.getNewFileName());

        // ── Activity 2: OCR ───────────────────────────────────────────────────
        // Receives result with {originalFileName, newFileName, localFilePath}
        // Returns result additionally with {extractedText}
        log.info("→ Step 2: Extracting text via OCR...");
        result = ocrActivity.extractText(result);
        log.info("← Step 2 complete: extracted {} chars",
                result.getExtractedText() != null ? result.getExtractedText().length() : 0);

        // ── Activity 3: Sentiment Analysis ────────────────────────────────────
        // Receives result with {originalFileName, newFileName, localFilePath, extractedText}
        // Returns result additionally with {sentimentResult}
        log.info("→ Step 3: Analyzing sentiment...");
        result = sentimentActivity.analyzeSentiment(result);
        log.info("← Step 3 complete: sentiment = '{}'", result.getSentimentResult());

        // ── ADD NEW ACTIVITY CALL HERE ────────────────────────────────────────
        // result = translationActivity.translate(result);

        log.info("═══ Workflow COMPLETE for '{}' — Sentiment: {} ═══",
                originalFileName, result.getSentimentResult());

        // Return fully-populated FileProcessResult to the REST Controller
        return result;
    }
}
