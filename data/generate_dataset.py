import numpy as np
import pandas as pd

rng = np.random.default_rng(77)

# =====================================================================
# Sheet 1: Items -- 60 products across 6 categories
# =====================================================================
catalog = {
    "Tech & Gadgets": [
        ("Wireless Earbuds Pro", "wireless bluetooth audio earbuds noise-cancelling", 59.99),
        ("Smart Fitness Band", "wearable tracker heart-rate steps sleep", 34.50),
        ("Portable Power Bank 20000mAh", "charger battery portable usb-c fast-charging", 28.00),
        ("Mechanical Keyboard", "keyboard mechanical rgb gaming typing", 79.99),
        ("4K Webcam", "webcam video camera streaming hd", 45.00),
        ("Bluetooth Speaker Mini", "speaker bluetooth portable audio waterproof", 22.99),
        ("Smart Plug 4-Pack", "smart-home plug wifi automation outlet", 24.00),
        ("USB-C Hub 7-in-1", "hub adapter usb-c ports dock", 32.50),
        ("Noise-Cancelling Headphones", "headphones noise-cancelling audio over-ear travel", 89.00),
        ("Phone Camera Lens Kit", "lens phone camera macro wide-angle", 19.99),
    ],
    "Outdoor & Camping": [
        ("2-Person Tent", "tent camping outdoor shelter lightweight", 65.00),
        ("Insulated Water Bottle", "bottle insulated water hiking steel", 18.50),
        ("Trekking Poles Pair", "poles hiking trekking adjustable aluminum", 27.00),
        ("Camping Hammock", "hammock camping outdoor portable nylon", 24.99),
        ("Headlamp Rechargeable", "headlamp light camping rechargeable led", 16.00),
        ("Portable Camp Stove", "stove camping cooking portable gas", 38.00),
        ("Sleeping Bag 3-Season", "sleeping-bag camping outdoor warm compact", 54.00),
        ("Waterproof Dry Bag 20L", "bag waterproof dry storage hiking kayaking", 21.50),
        ("Compact Binoculars", "binoculars outdoor hiking bird-watching travel", 42.00),
        ("Multi-Tool Knife", "tool knife camping multitool survival", 19.00),
    ],
    "Kitchen & Dining": [
        ("Stainless Steel Cookware Set", "cookware kitchen pots pans stainless", 120.00),
        ("Electric Kettle", "kettle electric kitchen tea coffee fast", 26.00),
        ("Non-Stick Frying Pan", "pan frying non-stick kitchen cooking", 22.00),
        ("Knife Sharpening Set", "knife sharpener kitchen tool set", 17.50),
        ("Glass Meal Prep Containers", "containers meal-prep glass kitchen storage", 29.99),
        ("Digital Kitchen Scale", "scale kitchen digital cooking baking", 14.00),
        ("Coffee Grinder Burr", "grinder coffee burr kitchen appliance", 34.00),
        ("Bamboo Cutting Board Set", "cutting-board bamboo kitchen prep eco", 19.50),
        ("Air Fryer 5L", "air-fryer kitchen appliance cooking healthy", 68.00),
        ("Ceramic Dinnerware Set", "dinnerware ceramic kitchen plates dining", 55.00),
    ],
    "Fitness & Wellness": [
        ("Yoga Mat Premium", "yoga mat fitness exercise non-slip", 24.00),
        ("Adjustable Dumbbell Set", "dumbbell weights fitness strength home-gym", 89.00),
        ("Resistance Bands Set", "resistance-bands fitness exercise home-workout", 15.00),
        ("Foam Roller", "foam-roller recovery fitness muscle massage", 18.50),
        ("Jump Rope Speed", "jump-rope fitness cardio exercise speed", 9.99),
        ("Massage Gun Percussion", "massage-gun recovery muscle fitness percussion", 59.00),
        ("Weighted Vest 10kg", "weighted-vest fitness training strength cardio", 44.00),
        ("Pilates Ring", "pilates-ring fitness toning exercise core", 16.00),
        ("Electrolyte Supplement Pack", "electrolyte supplement hydration fitness wellness", 21.00),
        ("Sleep & Recovery Pillow", "pillow sleep wellness recovery ergonomic", 32.00),
    ],
    "Books & Media": [
        ("The Midnight Orchard (Novel)", "novel fiction book drama mystery", 14.99),
        ("Mastering Personal Finance", "book finance money investing guide", 19.99),
        ("Science of Habit Formation", "book psychology self-help habits science", 16.50),
        ("World History Illustrated", "book history illustrated reference education", 24.00),
        ("Beginner's Guide to Coding", "book coding programming beginner technology", 22.50),
        ("Poetry of Quiet Mornings", "poetry book literature reflective calm", 12.00),
        ("The Startup Playbook", "book business startup entrepreneurship guide", 18.00),
        ("Mindful Cooking at Home", "book cooking recipes mindful lifestyle", 21.00),
        ("Astronomy for Beginners", "book astronomy science space beginner", 17.50),
        ("Graphic Novel: Steel Horizon", "graphic-novel comic fiction adventure illustrated", 15.00),
    ],
    "Office & Stationery": [
        ("Ergonomic Office Chair", "chair office ergonomic desk comfort", 145.00),
        ("Standing Desk Converter", "desk standing office ergonomic adjustable", 98.00),
        ("Fountain Pen Set", "pen fountain stationery writing gift", 28.00),
        ("Leather Notebook A5", "notebook leather stationery journal writing", 16.00),
        ("Desk Organizer Bamboo", "organizer desk bamboo office storage", 19.00),
        ("Whiteboard Calendar Planner", "planner whiteboard office organization calendar", 23.50),
        ("Wireless Mouse Ergonomic", "mouse wireless ergonomic office computer", 18.99),
        ("Desk Lamp LED Dimmable", "lamp desk led office lighting dimmable", 27.00),
        ("Sticky Notes Variety Pack", "sticky-notes stationery office organization", 7.50),
        ("Laptop Stand Adjustable", "stand laptop office ergonomic adjustable", 25.00),
    ],
}

rows = []
item_id = 1
for category, products in catalog.items():
    for title, tags, price in products:
        description = f"{title}. {tags.replace('-', ' ')}. Category: {category}."
        rows.append({
            "item_id": item_id, "title": title, "category": category,
            "tags": tags, "price": price, "description": description,
        })
        item_id += 1

items_df = pd.DataFrame(rows)
categories = list(catalog.keys())

# =====================================================================
# Sheet 2: Ratings -- 300 users, 3,460 ratings, 1-5 scale, with timestamps
# =====================================================================
N_USERS = 300
N_RATINGS = 3460

# Each user has a primary category (likes it, rates it higher) and a
# secondary category (mild interest), plus occasional ratings elsewhere.
# This gives both content-based and collaborative filtering real signal
# to find, rather than pure noise.
user_primary = rng.choice(categories, N_USERS)
user_secondary = np.array([
    rng.choice([c for c in categories if c != prim]) for prim in user_primary
])

dates = pd.date_range("2025-01-01", "2025-09-30", freq="D")

rating_rows = []
pairs_seen = set()

# One item is deliberately held back from ever being rated, to give a
# genuine, naturally-occurring "item cold start" case for later discussion.
cold_start_item_id = int(items_df.iloc[-1]["item_id"])  # last item (Laptop Stand Adjustable)
ratable_items = items_df[items_df["item_id"] != cold_start_item_id]

attempts = 0
while len(rating_rows) < N_RATINGS and attempts < N_RATINGS * 20:
    attempts += 1
    u = rng.integers(1, N_USERS + 1)

    # One user is deliberately held to a single rating, to give a genuine
    # "user cold start" case for later discussion.
    if u == N_USERS and len([r for r in rating_rows if r["user_id"] == u]) >= 1:
        continue

    roll = rng.random()
    if roll < 0.55:
        cat_pool = ratable_items[ratable_items["category"] == user_primary[u - 1]]
        base_rating = rng.choice([4, 4, 5, 5, 5, 3], )
    elif roll < 0.80:
        cat_pool = ratable_items[ratable_items["category"] == user_secondary[u - 1]]
        base_rating = rng.choice([3, 4, 4, 5, 2])
    else:
        cat_pool = ratable_items
        base_rating = rng.choice([1, 2, 3, 4, 5])

    item = cat_pool.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
    key = (u, int(item["item_id"]))
    if key in pairs_seen:
        continue
    pairs_seen.add(key)

    rating = int(np.clip(base_rating + rng.integers(-1, 2), 1, 5))
    timestamp = rng.choice(dates)

    rating_rows.append({
        "user_id": u, "item_id": int(item["item_id"]), "rating": rating,
        "timestamp": pd.Timestamp(timestamp).strftime("%Y-%m-%d"),
    })

ratings_df = pd.DataFrame(rating_rows[:N_RATINGS])

# =====================================================================
# Write both sheets to one workbook
# =====================================================================
out_path = "recommendation_system_dataset_v2.xlsx"
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    items_df.to_excel(writer, sheet_name="Items", index=False)
    ratings_df.to_excel(writer, sheet_name="Ratings", index=False)

print("Items:", items_df.shape, "categories:", items_df["category"].nunique())
print("Ratings:", ratings_df.shape)
print("Unique users rated:", ratings_df["user_id"].nunique())
print("Unique items rated:", ratings_df["item_id"].nunique(), "(60 total -> cold-start item id:", cold_start_item_id, ")")
print("Ratings for user", N_USERS, "(cold-start user):", (ratings_df["user_id"] == N_USERS).sum())
print("written to", out_path)
