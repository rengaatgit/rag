package com.example.fileprocessor.controller;

import com.example.fileprocessor.config.TemporalConfig;
import com.example.fileprocessor.model.FileProcessResult;
import com.example.fileprocessor.workflow.FileWorkflow;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowOptions;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;

/**
 * ════════════════════════════════════════════════════════════════
 *  REST CONTROLLER — ONE-TIME SETUP
 * ════════════════════════════════════════════════════════════════
 *
 * ROLE: Pure TRIGGER — submits work to Temporal and waits for result.
 *       Contains ZERO business logic.
 *
 * ENDPOINT:
 *   PUT /api/files/process
 *   Params:
 *     fileName    → e.g. "invoice.pdf"
 *     fileContent → raw text content of the file
 *
 * EXAMPLE CURL:
 *   curl -X PUT "http://localhost:8080/api/files/process" \
 *        -d "fileName=invoice.pdf" \
 *        -d "fileContent=Invoice+%231234%0ATotal+due%3A+%241200"
 *
 * RESPONSE (JSON):
 *   {
 *     "originalFileName": "invoice.pdf",
 *     "newFileName":      "invoice_20240101123045789.pdf",
 *     "localFilePath":    "/tmp/uploads/invoice_20240101123045789.pdf",
 *     "extractedText":    "Invoice #1234\nTotal due: $1,200",
 *     "sentimentResult":  "NEUTRAL (confidence: 0.88)"
 *   }
 *
 * @RequiredArgsConstructor → Lombok generates constructor that injects WorkflowClient
 */
@Slf4j
@RestController
@RequestMapping("/api/files")
@RequiredArgsConstructor
public class FileController {

    /**
     * WorkflowClient is injected from TemporalConfig.
     * This is the controller's ONLY connection to Temporal.
     */
    private final WorkflowClient workflowClient;

    /**
     * PUT /api/files/process
     *
     * What happens when this is called:
     *   1. Generate a unique workflowId for this file processing run
     *   2. Create WorkflowOptions (which queue, which workflow ID)
     *   3. Create a typed stub (proxy) for FileWorkflow
     *   4. Call processFile() on the stub → Temporal routes this to the Worker
     *   5. Worker executes Activity 1 → 2 → 3 sequentially
     *   6. Return the complete FileProcessResult as JSON
     *
     * This call BLOCKS until all 3 activities complete.
     * Use WorkflowClient.start() instead if you want async (fire-and-forget).
     */
    @PutMapping("/process")
    public ResponseEntity<FileProcessResult> processFile(
            @RequestParam String fileName,
            @RequestParam String fileContent) {

        log.info("══ PUT /api/files/process received for file: '{}' ══", fileName);

        // ── Step 1: Generate unique Workflow ID ───────────────────────────────
        // Each file upload = one isolated Workflow Run in Temporal.
        // You can search this ID in Temporal UI to see exactly what happened.
        // Format: "file-process-" + UUID   e.g. "file-process-a1b2c3d4-e5f6-..."
        String workflowId = "file-process-" + UUID.randomUUID();
        log.info("Assigning Workflow ID: {}", workflowId);

        // ── Step 2: Configure this specific workflow run ──────────────────────
        WorkflowOptions options = WorkflowOptions.newBuilder()
                .setWorkflowId(workflowId)
                // Must match TemporalConfig.TASK_QUEUE — worker polls this queue
                .setTaskQueue(TemporalConfig.TASK_QUEUE)
                .build();

        // ── Step 3: Create a TYPED workflow stub ──────────────────────────────
        // This is NOT a real FileWorkflow instance.
        // It's a Temporal PROXY — calling methods on it sends tasks to Temporal Server.
        // The actual FileWorkflowImpl runs in the Worker (possibly different thread/machine).
        FileWorkflow workflow = workflowClient.newWorkflowStub(
                FileWorkflow.class,
                options
        );

        // ── Step 4: Execute workflow (SYNCHRONOUS) ────────────────────────────
        // This single line triggers the entire pipeline:
        //   SaveFileActivity → OcrActivity → SentimentActivity
        //
        // It BLOCKS here until FileWorkflowImpl.processFile() returns.
        // During this time, you can watch progress in Temporal UI at:
        //   http://localhost:8080/namespaces/default/workflows/{workflowId}
        //
        // TO MAKE IT ASYNC (fire-and-forget):
        //   Replace this line with:
        //     WorkflowExecution execution = WorkflowClient.start(workflow::processFile, fileName, fileContent);
        //     return ResponseEntity.accepted().body(new FileProcessResult(fileName, null, null, null, null));
        FileProcessResult result = workflow.processFile(fileName, fileContent);

        log.info("══ Workflow {} COMPLETE. Sentiment: {} ══",
                workflowId, result.getSentimentResult());

        log.info("Temporal UI link: http://localhost:8080/namespaces/default/workflows/{}",
                workflowId);

        return ResponseEntity.ok(result);
    }
}
