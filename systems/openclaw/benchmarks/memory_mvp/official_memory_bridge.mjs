#!/usr/bin/env node

import process from "node:process";

import memoryCore, {
  buildPromptSection,
} from "../../dist/extensions/memory-core/index.js";
import {
  getMemorySearchManager,
} from "../../dist/plugin-sdk/memory-core.js";
import * as embeddingRegistry from "../../dist/memory-embedding-providers--EvawAcK.js";

const registeredToolFactories = new Map();
let builtinsRegistered = false;

function ensureBuiltinsRegistered() {
  if (builtinsRegistered) {
    return;
  }
  const noop = () => {};
  const api = new Proxy(
    {
      registerMemoryEmbeddingProvider: (adapter) => embeddingRegistry.a(adapter),
      registerTool: (factory, options = {}) => {
        const names = Array.isArray(options.names)
          ? options.names
          : options.name
            ? [options.name]
            : [];
        for (const name of names) {
          if (typeof name === "string" && name.trim()) {
            registeredToolFactories.set(name.trim(), factory);
          }
        }
      },
    },
    {
      get(target, prop) {
        return target[prop] ?? noop;
      },
    },
  );
  memoryCore.register(api);
  builtinsRegistered = true;
}

async function readJsonStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk)));
  }
  const text = Buffer.concat(chunks).toString("utf8").trim();
  return text ? JSON.parse(text) : {};
}

function writeJson(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

function serializeError(error) {
  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
      stack: error.stack,
    };
  }
  return {
    name: "Error",
    message: String(error),
  };
}

function extractToolPayload(result) {
  if (result && typeof result === "object" && result.details && typeof result.details === "object") {
    return result.details;
  }
  const text = Array.isArray(result?.content)
    ? result.content
        .filter((item) => item && typeof item === "object" && item.type === "text")
        .map((item) => String(item.text ?? ""))
        .join("\n")
        .trim()
    : "";
  if (!text) {
    return {};
  }
  try {
    return JSON.parse(text);
  } catch {
    return { text };
  }
}

function instantiateRegisteredTool(name, input) {
  const factory = registeredToolFactories.get(name);
  if (!factory) {
    throw new Error(`unregistered tool: ${name}`);
  }
  const tool = typeof factory === "function"
    ? factory({
        config: input.cfg,
        sessionKey: input.sessionKey,
      })
    : factory;
  if (!tool || typeof tool !== "object" || typeof tool.execute !== "function") {
    throw new Error(`tool unavailable: ${name}`);
  }
  return tool;
}

async function withManager(input, run) {
  const { cfg, agentId = "main" } = input;
  const { manager, error } = await getMemorySearchManager({
    cfg,
    agentId,
  });
  if (!manager) {
    throw new Error(error || "openclaw manager unavailable");
  }
  try {
    return await run(manager);
  } finally {
    await manager.close?.();
  }
}

async function main() {
  ensureBuiltinsRegistered();
  const input = await readJsonStdin();
  if (input.stateDir) {
    process.env.OPENCLAW_STATE_DIR = String(input.stateDir);
  }

  switch (String(input.op || "")) {
    case "sync": {
      const payload = await withManager(input, async (manager) => {
        await manager.sync({
          reason: "bridge-sync",
          force: Boolean(input.force),
        });
        return {
          ok: true,
          status: manager.status(),
        };
      });
      writeJson(payload);
      return;
    }
    case "status": {
      const payload = await withManager(input, async (manager) => ({
        ok: true,
        status: manager.status(),
      }));
      writeJson(payload);
      return;
    }
    case "search": {
      const tool = instantiateRegisteredTool("memory_search", input);
      const toolResult = await tool.execute("bridge-memory-search", {
        query: String(input.query || ""),
        maxResults:
          input.maxResults === null || input.maxResults === undefined
            ? undefined
            : Number(input.maxResults),
        minScore:
          input.minScore === null || input.minScore === undefined
            ? undefined
            : Number(input.minScore),
      });
      const payload = extractToolPayload(toolResult);
      writeJson({
        ok: true,
        payload,
      });
      return;
    }
    case "read_file": {
      const tool = instantiateRegisteredTool("memory_get", input);
      const toolResult = await tool.execute("bridge-memory-get", {
        path: String(input.path || ""),
        from:
          input.fromLine === null || input.fromLine === undefined
            ? undefined
            : Number(input.fromLine),
        lines:
          input.lines === null || input.lines === undefined
            ? undefined
            : Number(input.lines),
      });
      writeJson({
        ok: true,
        payload: extractToolPayload(toolResult),
      });
      return;
    }
    case "prompt_section": {
      const availableTools = new Set(
        Array.isArray(input.availableTools)
          ? input.availableTools
              .filter((name) => typeof name === "string")
              .map((name) => String(name))
          : ["memory_search", "memory_get"],
      );
      writeJson({
        ok: true,
        payload: {
          lines: buildPromptSection({
            availableTools,
            citationsMode: input.citationsMode ? String(input.citationsMode) : undefined,
          }),
        },
      });
      return;
    }
    default:
      throw new Error(`unsupported op: ${String(input.op || "")}`);
  }
}

main().catch((error) => {
  writeJson({
    ok: false,
    error: serializeError(error),
  });
  process.exitCode = 1;
});
