package com.example.fileprocessor.activity;

import com.example.fileprocessor.model.FileProcessResult;
import io.temporal.activity.ActivityInterface;
import io.temporal.activity.ActivityMethod;

/**
 * ════════════════════════════════════════════════════════════════
 *  ACTIVITY 3 INTERFACE — Sentiment Analysis
 * ════════════════════════════════════════════════════════════════
 *
 * WHAT:   Declares the contract for "analyze sentiment from text" work unit.
 *
 * PATTERN — Takes the baton, adds its own data, completes the pipeline:
 *   INPUT:  FileProcessResult (extractedText filled by Activity 2)
 *   OUTPUT: Same FileProcessResult with sentimentResult now filled in
 *           → At this point, ALL fields in FileProcessResult are populated.
 *
 * This activity is responsible for:
 *   1. Reading extractedText from the result
 *   2. Calling an external Sentiment Analysis API
 *   3. Getting the sentiment label + confidence score
 *   4. Storing formatted result in result.sentimentResult
 *   5. Returning the fully-complete result baton
 *
 * ── HOW TO ADD A NEW ACTIVITY (template to follow) ─────────────
 *
 *   @ActivityInterface
 *   public interface TranslationActivity {
 *       @ActivityMethod
 *       FileProcessResult translate(FileProcessResult result);
 *   }
 *
 *   Then create TranslationActivityImpl.java with your logic.
 *   Register in TemporalConfig.java (one line).
 *   Add one line in FileWorkflowImpl.java to call it.
 * ────────────────────────────────────────────────────────────────
 */
@ActivityInterface
public interface SentimentActivity {

    @ActivityMethod
    FileProcessResult analyzeSentiment(FileProcessResult result);
}
