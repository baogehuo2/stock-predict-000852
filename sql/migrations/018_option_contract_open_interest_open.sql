SET @column_exists := (
    SELECT COUNT(*)
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'option_contract_daily_raw'
      AND COLUMN_NAME = 'open_interest_open'
);

SET @ddl := IF(
    @column_exists = 0,
    'ALTER TABLE option_contract_daily_raw ADD COLUMN open_interest_open DECIMAL(24,4) NULL AFTER volume',
    'SELECT 1'
);

PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
