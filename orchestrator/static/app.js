"use strict";

// Everything shown here can come from the model or from customers, so it is only ever
// inserted as text (textContent, text nodes). Never build HTML from it.

const TOOL_LABELS = {
  get_order: "Looking up the order",
  list_orders: "Searching orders",
  preview_action: "Preparing a proposal",
};

const SUGGESTIONS = [
  "Which paid orders have not shipped in 3 days?",
  "Show me order 1042",
  "Please refund order 1042",
  "Show me order 2001",
];

const account = document.getElementById("account");
const modelPicker = document.getElementById("model-picker");
const model = document.getElementById("model");
const chat = document.getElementById("chat");
const messages = document.getElementById("messages");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const send = document.getElementById("send");
const hint = document.getElementById("hint");

let threadId = newThreadId();
let busy = false; // a turn is streaming
let waiting = false; // a proposal needs a decision

function newThreadId() {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollToEnd() {
  chat.scrollTop = chat.scrollHeight;
}

function add(node) {
  messages.querySelector(".empty")?.remove();
  messages.append(node);
  scrollToEnd();
  return node;
}

function refreshComposer() {
  const locked = busy || waiting;
  input.disabled = locked;
  send.disabled = locked;
  account.disabled = busy;
  model.disabled = busy;
  hint.textContent = waiting
    ? "Approve or reject the proposal to continue."
    : busy
      ? "Working..."
      : "";
  if (!locked) input.focus();
}

function appendInline(node, text) {
  for (const part of text.split(/(\*\*[^*]+\*\*|\*[^*\s][^*]*\*)/)) {
    if (/^\*\*[^*]+\*\*$/.test(part)) node.append(el("strong", null, part.slice(2, -2)));
    else if (/^\*[^*\s][^*]*\*$/.test(part)) node.append(el("em", null, part.slice(1, -1)));
    else if (part) node.append(document.createTextNode(part));
  }
}

// Handles the small part of Markdown that models use: paragraphs, bullets, bold, italics.
function renderAnswer(node, text) {
  node.replaceChildren();
  let list = null;
  for (const line of text.split("\n")) {
    const bullet = line.match(/^\s*[*-]\s+(.*)$/);
    if (bullet) {
      if (!list) list = node.appendChild(el("ul"));
      const item = list.appendChild(el("li"));
      appendInline(item, bullet[1]);
    } else {
      list = null;
      if (line.trim()) appendInline(node.appendChild(el("p")), line);
    }
  }
}

function showEmptyState() {
  const empty = el("div", "empty");
  empty.append(el("h2", null, "How can I help with your orders?"));
  empty.append(el("p", null, "Refunds and cancellations always wait for your approval."));
  const list = empty.appendChild(el("div", "suggestions"));
  for (const text of SUGGESTIONS) {
    const button = list.appendChild(el("button", "suggestion", text));
    button.type = "button";
    button.addEventListener("click", () => sendMessage(text));
  }
  messages.append(empty);
}

function startConversation() {
  threadId = newThreadId();
  waiting = false;
  messages.replaceChildren();
  showEmptyState();
  refreshComposer();
}

async function* readEvents(response) {
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value;
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop();
    for (const block of blocks) {
      let name = null;
      let data = null;
      for (const line of block.split(/\r?\n/)) {
        if (line.startsWith("event:")) name = line.slice(6).trim();
        else if (line.startsWith("data:")) data = line.slice(5).trim();
      }
      if (name && data) yield { name, data: JSON.parse(data) };
    }
  }
}

async function errorMessage(response) {
  try {
    const { detail } = await response.json();
    if (typeof detail === "string") return detail;
  } catch {
    // Not JSON: fall through to the general message.
  }
  return `The request failed (HTTP ${response.status}).`;
}

function addToolRow(activity, tool) {
  const row = activity.appendChild(el("div", "activity-row"));
  row.dataset.tool = tool;
  row.append(el("span", "spinner"), el("span", null, TOOL_LABELS[tool] ?? `Running ${tool}`));
  scrollToEnd();
}

function finishToolRow(activity, tool) {
  const row = activity.querySelector(`[data-tool="${CSS.escape(tool)}"] .spinner`);
  if (!row) return;
  row.className = "check";
  row.textContent = "✓";
}

function addDetail(list, label, value, className) {
  list.append(el("dt", null, label), el("dd", className, value));
}

function addProposal(proposal) {
  waiting = true;
  const verb = proposal.action === "refund" ? "Refund" : "Cancel";
  const card = el("section", "proposal");
  card.append(el("div", "proposal-label", "Needs your approval"));
  card.append(el("h2", null, `${verb} order ${proposal.order_id}`));

  const details = card.appendChild(el("dl"));
  addDetail(details, "Customer", proposal.customer_name);
  addDetail(details, "Store", proposal.merchant);
  addDetail(details, "Status", `${proposal.current_status} → ${proposal.new_status}`);
  addDetail(details, "Order total", proposal.order_total.toFixed(2));
  addDetail(details, "Back to customer", proposal.amount_to_customer.toFixed(2), "amount");

  const actions = card.appendChild(el("div", "proposal-actions"));
  const approve = actions.appendChild(el("button", "button primary", "Approve"));
  const reject = actions.appendChild(el("button", "button ghost", "Reject"));
  const status = card.appendChild(el("p", "proposal-status"));
  status.hidden = true;

  async function decide(decision) {
    approve.disabled = reject.disabled = true;
    status.hidden = false;
    status.textContent = decision === "approve" ? "Approving..." : "Rejecting...";
    waiting = false;
    const ok = await streamTurn(`/chat/${threadId}/${decision}`);
    if (ok) {
      actions.remove();
      card.classList.add(`decided-${decision}`);
      status.textContent = decision === "approve" ? "Approved" : "Rejected";
    } else {
      // Nothing was decided, so the card stays usable.
      waiting = true;
      approve.disabled = reject.disabled = false;
      status.hidden = true;
      refreshComposer();
    }
  }

  approve.addEventListener("click", () => decide("approve"));
  reject.addEventListener("click", () => decide("reject"));
  add(card);
}

// Streams one turn. Returns true if it ended normally.
async function streamTurn(path, body) {
  busy = true;
  refreshComposer();
  const activity = add(el("div", "activity"));
  const thinking = activity.appendChild(el("div", "activity-row"));
  thinking.append(el("span", "spinner"), el("span", null, "Thinking"));

  let bubble = null;
  let answer = "";
  let finished = false;
  let ok = false;

  try {
    const headers = { Authorization: `Bearer ${account.value}` };
    if (body) headers["Content-Type"] = "application/json";
    const response = await fetch(path, {
      method: "POST",
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!response.ok) {
      add(el("div", "notice", await errorMessage(response)));
      return false;
    }
    for await (const { name, data } of readEvents(response)) {
      thinking.remove();
      if (name === "tool_started") {
        addToolRow(activity, data.tool);
      } else if (name === "tool_finished") {
        finishToolRow(activity, data.tool);
      } else if (name === "token") {
        bubble ??= add(el("div", "bubble assistant"));
        answer += data.text;
        renderAnswer(bubble, answer);
        scrollToEnd();
      } else if (name === "proposal") {
        finished = ok = true;
        addProposal(data);
      } else if (name === "done") {
        finished = ok = true;
      } else if (name === "error") {
        finished = true;
        add(el("div", "notice", data.message));
      }
    }
    if (!finished) add(el("div", "notice", "The connection ended before the reply finished."));
    return ok;
  } catch {
    add(el("div", "notice", "Could not reach the assistant. Is it running?"));
    return false;
  } finally {
    thinking.remove();
    if (!activity.childElementCount) activity.remove();
    busy = false;
    refreshComposer();
  }
}

function sendMessage(text) {
  const message = text.trim();
  if (!message || busy || waiting) return;
  add(el("div", "bubble user", message));
  input.value = "";
  input.style.height = "";
  const body = { message, thread_id: threadId };
  if (!modelPicker.hidden) body.model = model.value;
  streamTurn("/chat", body);
}

async function loadModels() {
  try {
    const response = await fetch("/models");
    const { models } = await response.json();
    model.replaceChildren(...models.map(({ id, label }) => new Option(label, id)));
    // With one model there is nothing to choose.
    modelPicker.hidden = models.length < 2;
  } catch {
    modelPicker.hidden = true;
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(input.value);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage(input.value);
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight + 2}px`;
});

document.getElementById("new-chat").addEventListener("click", () => {
  if (!busy) startConversation();
});

// A conversation belongs to one store, so switching account starts a new one.
account.addEventListener("change", startConversation);

startConversation();
loadModels();
