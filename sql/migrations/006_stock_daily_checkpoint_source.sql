UPDATE stock_daily_collection_checkpoint
SET data_source = 'akshare:stock_daily_sina_eastmoney'
WHERE data_source = 'akshare:stock_zh_a_daily';
