ALTER TABLE margin_market_daily
    ADD COLUMN securities_lending_remaining_volume DECIMAL(24,4)
    AFTER securities_lending_sell_volume;

