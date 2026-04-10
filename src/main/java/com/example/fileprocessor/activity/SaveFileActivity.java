package com.example.fileprocessor.activity;

import com.example.fileprocessor.model.FileProcessResult;
import io.temporal.activity.ActivityInterface;
import io.temporal.activity.ActivityMethod;

/**
 * ════════════════════════════════════════════════════════════════
 *  ACTIVITY 1 INTERFACE — Save File Locally
 * ════════════════════════════════════════════════════════════════
 *
 * WHAT:   Declares the contract for "save file" work unit.
 * WHO:    Temporal uses this interface to route tasks to SaveFileActivityImpl.
 * WHY:    Separating interface from impl enables:
 *           - Easy mocking in tests
 *           - Temporal can intercept calls and add retry/timeout behavior
 *           - Multiple implementations possible (e.g. save to S3 vs local disk)
 *
 * @ActivityInterface → Marks this as a Temporal activity declaration.
 *                      Temporal will scan for this annotation during registration.
 *
 * @ActivityMethod    → Marks the method that Temporal will schedule and track.
 *                      Appears as a named EVENT in Temporal UI event history.
 *
 * INPUT:
 *   originalFileName  → e.g. "invoice.pdf"
 *   fileContent       → raw text content of the file
 *
 * OUTPUT:
 *   FileProcessResult with originalFileName + newFileName + localFilePath filled.
 *   (extractedText and sentimentResult will be null at this stage)
 */
@ActivityInterface
public interface SaveFileActivity {

    @ActivityMethod
    FileProcessResult saveFile(String originalFileName, String fileContent);
}
