"""Task 1 - seeded, deterministic order generator + validator."""
import random, collections

SEED = 7   # seed 42 gave 8.33% delayed (outside 10-30%), so seed changed per brief
N_ORDERS = 60
CATEGORIES = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
CATEGORY_WEIGHTS = [0.25, 0.15, 0.15, 0.20, 0.25]
STATUSES = ["Placed", "Shipped", "Delivered", "Returned", "Refunded"]
STATUS_WEIGHTS = [0.15, 0.25, 0.35, 0.15, 0.10]
DELAY_PROB = 0.20
# Reasoning: INR ranges mirror typical Nykaa baskets - low for beauty/apparel, high for electronics.
PRICE_RANGE = {"Apparel": (499, 4999), "Electronics": (1999, 59999), "Home": (799, 14999),
               "Footwear": (999, 7999), "Beauty": (199, 3999)}


def generate_orders(seed=SEED, n=N_ORDERS):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        cat = rng.choices(CATEGORIES, CATEGORY_WEIGHTS)[0]
        lo, hi = PRICE_RANGE[cat]
        out.append({"record_id": f"NYK-{1001 + i}", "category": cat,
                    "status": rng.choices(STATUSES, STATUS_WEIGHTS)[0],
                    "order_value_inr": rng.randint(lo, hi),
                    "days_since_created": rng.randint(0, 30),
                    "delayed_shipment": rng.random() < DELAY_PROB})
    return out


def stats(orders):
    cc = collections.Counter(o["category"] for o in orders)
    sc = collections.Counter(o["status"] for o in orders)
    pct = 100 * sum(o["delayed_shipment"] for o in orders) / len(orders)
    return dict(cc), dict(sc), round(pct, 2)


def validate(orders):
    cc, sc, pct = stats(orders)
    assert len(orders) >= 40, "need >= 40 orders"
    assert len({o["record_id"] for o in orders}) == len(orders), "duplicate ids"
    assert all(cc.get(c, 0) >= 3 for c in CATEGORIES), f"category counts too low: {cc}"
    assert all(sc.get(s, 0) >= 1 for s in STATUSES), f"status missing: {sc}"
    assert 10 <= pct <= 30, f"delayed % {pct} outside 10-30"
    assert all(0 <= o["days_since_created"] <= 30 for o in orders)
    return cc, sc, pct


ORDERS = generate_orders()
ORDER_INDEX = {o["record_id"]: o for o in ORDERS}

if __name__ == "__main__":
    cc, sc, pct = validate(ORDERS)
    print(f"seed={SEED} n={len(ORDERS)}\nPer category: {cc}\nPer status:   {sc}\nDelayed: {pct}%  (band 10-30%) -> OK")
    print("Sample:", ORDERS[0])
