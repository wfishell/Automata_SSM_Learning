import random
from collections import defaultdict

random.seed(42)

# Config
NUM_GPUS = 8
NUM_CUSTOMERS = 4
MAX_OCCUPANCY = 0.25  # 25% cap
MAX_PER_CUSTOMER = int(NUM_GPUS * MAX_OCCUPANCY)  # = 2 GPUs
GRANT_DURATION = 1  # hours
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Track active grants: list of (customer, expiry_hour)
active_grants = []


# Track holdings per customer
def get_holdings():
    holdings = defaultdict(int)
    for cust, _ in active_grants:
        holdings[cust] += 1
    return holdings


def total_allocated():
    return len(active_grants)


# Expire grants
def expire_grants(current_hour):
    global active_grants
    active_grants = [(c, exp) for c, exp in active_grants if exp > current_hour]


# Generate requests for an hour
def generate_requests(day_idx, hour):
    requests = []

    # C1 only requests between 12am-5am, but very aggressively
    if 0 <= hour <= 5:
        # C1 always requests, and often requests multiple times
        requests.append(1)
        if random.random() < 0.7:  # 70% chance of second request
            requests.append(1)
        if random.random() < 0.4:  # 40% chance of third request
            requests.append(1)

    # C2, C3, C4 request during daytime (8am-9pm) with higher frequency
    if 8 <= hour <= 21:
        for c in [2, 3, 4]:
            if random.random() < 0.5:  # 50% chance each
                requests.append(c)
            if random.random() < 0.3:  # 30% chance of second request
                requests.append(c)

    return requests


# Process requests with round-robin priority
def process_requests(requests, current_hour):
    results = []

    # Sort by priority: C1 > C2 > C3 > C4
    requests_sorted = sorted(requests)

    for cust in requests_sorted:
        holdings = get_holdings()

        if holdings[cust] >= MAX_PER_CUSTOMER:
            results.append((cust, "Reject (25%)"))
        elif total_allocated() >= NUM_GPUS:
            results.append((cust, "Reject (full)"))
        else:
            active_grants.append((cust, current_hour + GRANT_DURATION))
            results.append((cust, "Grant"))

    return results


# Generate the week
log = []
global_hour = 0

for day_idx, day in enumerate(DAYS):
    for hour in range(24):
        expire_grants(global_hour)

        requests = generate_requests(day_idx, hour)

        if requests:
            results = process_requests(requests, global_hour)

            for cust, action in results:
                holdings = get_holdings()
                h_str = f"{holdings[1]}/{holdings[2]}/{holdings[3]}/{holdings[4]}"
                total = total_allocated()
                log.append((day, hour, f"C{cust}", action, h_str, total))

        global_hour += 1

# Print as LaTeX table rows
for day, hour, cust, action, holdings, total in log:
    action_str = f"\\textbf{{{action}}}" if "Reject" in action else action
    print(f"{day} & {hour:02d}:00 & {cust} & {action_str} & {holdings} & {total} \\\\")

# Print summary stats
grants = sum(1 for _, _, _, action, _, _ in log if action == "Grant")
rejects = sum(1 for _, _, _, action, _, _ in log if "Reject" in action)
print(
    f"\n% Summary: {grants} grants, {rejects} rejects ({100*rejects/(grants+rejects):.1f}% rejection rate)"
)
