package com.example.fileprocessor.config;

import com.example.fileprocessor.activity.OcrActivityImpl;
import com.example.fileprocessor.activity.SaveFileActivityImpl;
import com.example.fileprocessor.activity.SentimentActivityImpl;
import com.example.fileprocessor.workflow.FileWorkflowImpl;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.serviceclient.WorkflowServiceStubs;
import io.temporal.serviceclient.WorkflowServiceStubsOptions;
import io.temporal.worker.Worker;
import io.temporal.worker.WorkerFactory;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * ════════════════════════════════════════════════════════════════
 *  TEMPORAL CONFIGURATION — ONE-TIME SETUP
 *  The "wiring" file that connects everything together.
 * ════════════════════════════════════════════════════════════════
 *
 * WHAT THIS FILE DOES (plain English):
 *   1. Opens a connection to Temporal Server  (like JDBC connects to a database)
 *   2. Creates a WorkflowClient               (like a SessionFactory in Hibernate)
 *   3. Creates a WorkerFactory                (manages worker lifecycle)
 *   4. Creates a Worker bound to TASK_QUEUE   (the executor that runs our code)
 *   5. Registers Workflow + Activity impls    (tells Worker what it can execute)
 *   6. Starts the Worker                      (begins polling for tasks)
 *
 * ══ TASK QUEUE CONCEPT ══════════════════════════════════════════
 *
 *   TASK_QUEUE = "file-processing-queue"
 *
 *   Think of it as a named channel:
 *     FileController  ──PUSH──▶  [file-processing-queue]  ◀──POLL──  Worker
 *     (sends work)                  (Temporal Server)               (executes work)
 *
 *   The name must match in BOTH:
 *     - TemporalConfig (where Worker polls from)
 *     - FileController (where WorkflowOptions sets the task queue)
 *
 * ══ WHEN TO MODIFY THIS FILE ════════════════════════════════════
 *   → Adding a new Activity: add the impl as a parameter + register it (2 lines)
 *   → Everything else: never needs to change
 *
 * ══ HOW TO ADD A NEW ACTIVITY ════════════════════════════════════
 *   Example: Adding TranslationActivity
 *
 *   Step 1: Add parameter to fileWorker():
 *     TranslationActivityImpl translationActivity
 *
 *   Step 2: Add to registerActivitiesImplementations():
 *     worker.registerActivitiesImplementations(
 *         saveFileActivity, ocrActivity, sentimentActivity,
 *         translationActivity   ← ADD HERE
 *     );
 *
 *   That's it! Spring auto-injects it. Temporal routes tasks to it.
 * ════════════════════════════════════════════════════════════════
 */
@Slf4j
@Configuration
public class TemporalConfig {

    /**
     * The task queue name — shared constant used by both Worker and Controller.
     * Must be identical in FileController's WorkflowOptions.
     */
    public static final String TASK_QUEUE = "file-processing-queue";

    @Value("${spring.temporal.connection.target:127.0.0.1:7233}")
    private String temporalTarget;

    @Value("${spring.temporal.namespace:default}")
    private String namespace;

    /**
     * BEAN 1 — WorkflowServiceStubs
     *
     * The raw gRPC connection to the Temporal Server.
     * Analogy: Like a JDBC Driver — low-level connection plumbing.
     *
     * For LOCAL development: connects to localhost:7233
     * For PRODUCTION: change temporalTarget in application.properties
     *   spring.temporal.connection.target=temporal.yourcompany.com:7233
     */
    @Bean
    public WorkflowServiceStubs workflowServiceStubs() {
        log.info("Connecting to Temporal Server at: {}", temporalTarget);
        return WorkflowServiceStubs.newServiceStubs(
                WorkflowServiceStubsOptions.newBuilder()
                        .setTarget(temporalTarget)
                        .build()
        );
    }

    /**
     * BEAN 2 — WorkflowClient
     *
     * High-level client for interacting with Temporal:
     *   - Start new workflow runs
     *   - Query workflow state
     *   - Send signals to running workflows
     *   - Wait for workflow completion
     *
     * Analogy: Like EntityManagerFactory in JPA — the main entry point.
     *
     * This bean is @Autowired into FileController to trigger workflow runs.
     */
    @Bean
    public WorkflowClient workflowClient(WorkflowServiceStubs stubs) {
        return WorkflowClient.newInstance(
                stubs,
                WorkflowClientOptions.newBuilder()
                        .setNamespace(namespace)
                        // Namespace = logical isolation (like a database schema)
                        // "default" is fine for development
                        .build()
        );
    }

    /**
     * BEAN 3 — WorkerFactory
     *
     * Manages the lifecycle of all Worker instances.
     * Handles thread pools, graceful shutdown, etc.
     *
     * Analogy: Like an ExecutorService factory — creates and manages workers.
     */
    @Bean
    public WorkerFactory workerFactory(WorkflowClient workflowClient) {
        return WorkerFactory.newInstance(workflowClient);
    }

    /**
     * BEAN 4 — Worker
     *
     * The actual executor that:
     *   1. Polls TASK_QUEUE for pending workflow/activity tasks
     *   2. Deserializes task inputs from JSON
     *   3. Calls the registered implementation (e.g. SaveFileActivityImpl)
     *   4. Serializes return value to JSON
     *   5. Reports result back to Temporal Server
     *
     * This bean starts automatically when Spring context loads.
     * It runs as a background thread — always listening.
     *
     * In Temporal UI → "Workers" tab: you'll see this worker's identity,
     * task queue, registered workflow types, and registered activity types.
     *
     * ══ ADD NEW ACTIVITY IMPL PARAMETER + REGISTRATION HERE ════
     */
    @Bean
    public Worker fileWorker(
            WorkerFactory factory,
            // Spring injects these @Component beans automatically
            SaveFileActivityImpl saveFileActivity,
            OcrActivityImpl ocrActivity,
            SentimentActivityImpl sentimentActivity
            // STEP 1 of 2 when adding new activity → add parameter here:
            // TranslationActivityImpl translationActivity
    ) {
        // Create a Worker bound to our specific task queue
        Worker worker = factory.newWorker(TASK_QUEUE);

        // ── Register Workflow Implementation ──────────────────────────────────
        // Tells Temporal: "This worker can execute FileWorkflow tasks"
        // Note: Pass the CLASS, not an instance — Temporal creates new instances per run
        worker.registerWorkflowImplementationTypes(FileWorkflowImpl.class);

        // ── Register Activity Implementations ─────────────────────────────────
        // Tells Temporal: "This worker can execute these specific activities"
        // Note: Pass INSTANCES (not classes) — activities are reused across calls
        worker.registerActivitiesImplementations(
                saveFileActivity,
                ocrActivity,
                sentimentActivity
                // STEP 2 of 2 when adding new activity → add instance here:
                // translationActivity
        );

        // Start polling — Worker begins listening on TASK_QUEUE immediately
        factory.start();

        log.info("✅ Temporal Worker started. Polling queue: '{}'", TASK_QUEUE);
        log.info("   Temporal UI: http://localhost:8080/namespaces/{}/workflows", namespace);

        return worker;
    }
}
