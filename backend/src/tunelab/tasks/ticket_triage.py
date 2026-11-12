"""Support-ticket triage: free-text ticket -> strict JSON record.

The dataset is SYNTHETIC and generated deterministically from a seed. Labels are not inferred
after the fact: every label is a generation parameter, and the ticket text is rendered from that
label with template phrases, so ground truth is exact. Templates are split into "seen" and
"held-out" sets; the held-out test split uses only phrasings never shown during training, so it
measures generalisation beyond memorised wording.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass

CATEGORIES = ["billing", "technical", "account", "shipping", "feature_request"]
PRIORITIES = ["low", "medium", "high", "urgent"]
PRODUCTS = ["Atlas Router", "Nimbus Backup", "Pulse Thermostat", "Orbit Mobile App", "Vertex Analytics"]
FIELDS = ["category", "priority", "product", "order_id", "refund_requested"]

SYSTEM_PROMPT = (
    "You triage customer support tickets. Reply with ONLY a JSON object with exactly these keys:\n"
    '"category": one of "billing", "technical", "account", "shipping", "feature_request";\n'
    '"priority": one of "low", "medium", "high", "urgent";\n'
    '"product": one of "Atlas Router", "Nimbus Backup", "Pulse Thermostat", "Orbit Mobile App", "Vertex Analytics";\n'
    '"order_id": the order id like "ORD-12345" if the ticket mentions one, otherwise null;\n'
    '"refund_requested": true if the customer asks for a refund or money back, otherwise false.'
)


@dataclass(frozen=True)
class Label:
    category: str
    priority: str
    product: str
    order_id: str | None
    refund_requested: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(", ", ": "))


# Category bodies. {p} = product, {o} = order clause. Index >= SEEN_PER_CATEGORY are held out.
BODIES: dict[str, list[str]] = {
    "billing": [
        "I was charged twice for my {p} subscription this month{o}.",
        "My invoice for {p} shows a price higher than what I signed up for{o}.",
        "The card payment for {p} keeps failing even though my bank says it is fine{o}.",
        "I cancelled {p} last week but I was still billed{o}.",
        "Can someone explain the extra fee on my {p} bill{o}?",
        "There is a charge from you for {p} that I do not recognise{o}.",
        "You billed me the annual {p} plan instead of the monthly one{o}.",
        "My {p} receipt lists the wrong tax amount{o}.",
    ],
    "technical": [
        "The {p} keeps disconnecting every few minutes{o}.",
        "{p} crashes as soon as I open the settings page{o}.",
        "After the latest update {p} will not start at all{o}.",
        "{p} shows an error code E-41 when syncing{o}.",
        "The {p} is very slow and sometimes freezes completely{o}.",
        "{p} stopped saving my changes since yesterday{o}.",
        "My {p} reboots by itself during the night{o}.",
        "{p} reports that my data is corrupted when I try to restore{o}.",
    ],
    "account": [
        "I cannot log in to my {p} account, the password reset email never arrives{o}.",
        "Please change the email address on my {p} account{o}.",
        "Someone else seems to be using my {p} account{o}.",
        "I want to delete my {p} account and all my data{o}.",
        "Two-factor codes for {p} are not accepted anymore{o}.",
        "How do I add a second user to our {p} account{o}?",
        "My {p} account got locked after I typed the wrong password{o}.",
        "I need to transfer my {p} account to a colleague{o}.",
    ],
    "shipping": [
        "My {p} has not arrived yet and tracking has not updated in a week{o}.",
        "The {p} box arrived damaged{o}.",
        "I received the wrong item instead of the {p} I ordered{o}.",
        "The courier says my {p} was delivered but I never got it{o}.",
        "Can I change the delivery address for my {p}{o}?",
        "My {p} order is stuck in customs{o}.",
        "Only half of my {p} order was in the package{o}.",
        "The delivery date for my {p} keeps moving back{o}.",
    ],
    "feature_request": [
        "It would be great if {p} supported dark mode.",
        "Please add an export to CSV option in {p}.",
        "Could {p} integrate with our calendar app?",
        "I would love scheduled reports in {p}.",
        "Any plans to let {p} work offline?",
        "Suggestion: let {p} group devices by room.",
        "It would help if {p} had keyboard shortcuts.",
        "Please consider adding a family plan for {p}.",
    ],
}
SEEN_PER_CATEGORY = 6  # bodies 0-5 seen in training, 6-7 held out

URGENCY: dict[str, list[str]] = {
    "urgent": [
        "This is a production outage for our whole company.",
        "This is a security issue and needs attention right now.",
        "Our entire team is blocked and losing money every hour.",
        "Everything is down, please treat this as an emergency.",
    ],
    "high": ["I cannot use it at all until this is fixed.", "This is blocking my work today.", "I need this resolved as soon as possible.", "This is a serious problem for me."],
    "medium": ["It is annoying but I can work around it.", "It happens a few times a week.", "Please look into it when you can this week.", "Not critical, but it is getting frustrating."],
    "low": ["No rush at all.", "Just letting you know, whenever you get a chance.", "This is a minor thing.", "Low priority, just curious."],
}
SEEN_URGENCY = 3  # phrases 0-2 seen, 3 held out

REFUND = ["I want a refund.", "Please give me my money back.", "I expect a full refund for this.", "Refund me please."]
SEEN_REFUND = 3
OPENERS = ["Hi,", "Hello support,", "Hey team,", "Good morning,", "", "To whom it may concern,"]
CLOSERS = ["Thanks, Sam", "Regards, Priya", "- Tom", "Thank you.", "", "Best, Alex"]

# Category-dependent label constraints keep the data plausible.
ALLOWED_PRIORITY = {"feature_request": ["low", "medium"]}
REFUND_CATEGORIES = {"billing", "shipping", "technical"}


def _order_clause(rng: random.Random, order_id: str | None) -> str:
    if not order_id:
        return ""
    return rng.choice([f" (order {order_id})", f", order number {order_id}", f" on order {order_id}", f" - my order is {order_id}"])


def make_example(rng: random.Random, heldout: bool) -> dict:
    category = rng.choice(CATEGORIES)
    priority = rng.choice(ALLOWED_PRIORITY.get(category, PRIORITIES))
    product = rng.choice(PRODUCTS)
    has_order = (category in {"billing", "shipping"} and rng.random() < 0.75) or (category == "technical" and rng.random() < 0.25)
    order_id = f"ORD-{rng.randint(10000, 99999)}" if has_order else None
    refund = category in REFUND_CATEGORIES and rng.random() < 0.35

    pick = (lambda items, seen: rng.choice(items[seen:])) if heldout else (lambda items, seen: rng.choice(items[:seen]))
    body = pick(BODIES[category], SEEN_PER_CATEGORY).format(p=product, o=_order_clause(rng, order_id))
    parts = [rng.choice(OPENERS), body, pick(URGENCY[priority], SEEN_URGENCY)]
    if refund:
        parts.append(pick(REFUND, SEEN_REFUND))
    parts.append(rng.choice(CLOSERS))
    text = " ".join(p for p in parts if p)
    label = Label(category, priority, product, order_id, refund)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": label.to_json()},
        ],
        "meta": {"heldout_templates": heldout},
    }


def generate(seed: int = 7, train: int = 600, valid: int = 80, test: int = 120, test_heldout: int = 120) -> dict[str, list[dict]]:
    """Deterministic splits. Train/valid/test use seen templates; test_heldout uses only unseen ones."""
    rng = random.Random(seed)
    seen: set[str] = set()

    def draw(n: int, heldout: bool) -> list[dict]:
        out: list[dict] = []
        while len(out) < n:
            ex = make_example(rng, heldout)
            key = ex["messages"][1]["content"]
            if key in seen:  # no exact duplicates across any split
                continue
            seen.add(key)
            out.append(ex)
        return out

    return {"train": draw(train, False), "valid": draw(valid, False), "test": draw(test, False), "test_heldout": draw(test_heldout, True)}
