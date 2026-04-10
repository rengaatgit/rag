package com.example.fileprocessor.workflow;

import com.example.fileprocessor.model.FileProcessResult;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;

/**
 * ════════════════════════════════════════════════════════════════
 *  WORKFLOW INTERFACE — ONE-TIME SETUP, rarely modified
 * ════════════════════════════════════════════════════════════════
 *
 * WHAT:   Declares the overall FILE PROCESSING PIPELINE as a single contract.
 *
 * WHO CALLS IT:
 *   FileController.java creates a stub of this interface and calls processFile().
 *   The stub is a Temporal proxy — the actual execution happens in FileWorkflowImpl
 *   running inside the Worker, potentially on a different machine/thread.
 *
 * @WorkflowInterface → Marks this as a Temporal workflow declaration.
 *                      Temporal generates a client-side proxy (stub) from this.
 *
 * @WorkflowMethod    → The entry point for the workflow.
 *                      Appears as the top-level "Workflow Run" in Temporal UI.
 *                      Only ONE @WorkflowMethod per @WorkflowInterface.
 *
 * INPUT:
 *   originalFileName  → e.g. "invoice.pdf"
 *   fileContent       → text content of the uploaded file
 *
 * OUTPUT:
 *   FileProcessResult with ALL fields filled:
 *     ✅ originalFileName
 *     ✅ newFileName
 *     ✅ localFilePath
 *     ✅ extractedText
 *     ✅ sentimentResult
 */
@WorkflowInterface
public interface FileWorkflow {

    @WorkflowMethod
    FileProcessResult processFile(String originalFileName, String fileContent);
}
