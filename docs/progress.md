# Reverb API Data Collection – Recap

## 🎯 Goals

- Build a datawarehouse to run analysis on the musical instruments market. Focus:
    - Used Instruments: 
        - monitor availability. Understand periods of higher availability and/or lower prices
          - Key questions: "is it a good time to buy/sell? What is it going to be?"
        - With Comparison with other sites, like `mercatinomusicale.net`:
          - "where to look?"
- Create a data visualization with tools like R, Tableau, PowerBI...

## ✅ Done
- **Found API documentation**
  - on UNPKG and openAPI: useful details about functioning

- **Explored API behavior**
  - Verified `Accept-Version: 3.0` and `Accept-Language: it-IT` headers.
  - Found `per_page` maximum = 50.
  - Learned `price` = seller baseline, `buyer_price` = buyer-context price.
  - Confirmed timestamps (`created_at`, `published_at`) are ISO 8601.
  - Understood `state.slug = live` = only live listings are returned.

- **Taxonomy & categories**
  - Downloaded JSON of categories.
  - Restructured into a CSV lookup table (product_type, category, listings_href).

- **Filtering**
  - Wrote a `filter_listing(x)` filter/flatten function.
  - Filter and flatten fields.
  - For now, **ignore shipping** (too complex; revisit later).

- **Counting phase**
  - Sequential benchmark script to fetch `total` per category with `per_page=1`.
  - Added polite delays + progress logging with tqdm.
  - Measured average response times.

## 📝 Decisions
- Work **sequentially first** → establish viability, measure times. 
- Store **both `price` and `buyer_price`**, but for *Italian stakeholders*:
  - use Italian context (IP + Accept-Language),
  - filter only listings that ship to Italy (later).
- Ignore shipping for now, but long-term: include only relevant shipping rates (XX/EU/IT).

## 🚧 To Do
- [ ] Understand how to get data localized to Italy, as in web browsing experience 
- [ ] Assemble **requester**: iterate categories, fetch pages (`per_page=50`).
- [ ] Apply **filtering function** to each listing, flatten → DataFrame/Parquet.
- [ ] Dedup by `id` (idempotent storage).
    - On `id`: if there are changes, update the record, registering time of update. If not present anymore, register status as closed and its time
- [ ] Add scheduling/rotation (some categories daily, others weekly), considering category numerosity to avoid 429 errors
- [ ] Define exact **scope of analysis**:
  - Seller view (listing `price`) vs Buyer view (Italian `buyer_price + shipping`).
- [ ] Revisit **shipping logic** once stakeholder scope is finalized.