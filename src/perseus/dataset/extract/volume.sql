SELECT 
    sp.signal_id,
    (sp.volume::JSONB ->> 'volume_traded_bc')::FLOAT AS volume_traded_bc,
    (sp.duration::JSONB ->> 'duration_min')::FLOAT AS duration_min,
	(sp.basic_metrics::JSONB ->> 'total_volume_bc')::FLOAT AS total_volume_bc,
	ss.commodity,
    ss.base_commodity,
    ss.id,
	ss.source_posted_at 
FROM 
    signals_pumpsignal sp
JOIN signals_signal ss
ON
    sp.id = ss.message_id
WHERE 
    sp.volume IS NOT NULL
    AND sp.duration IS NOT NULL
    AND sp.volume ~ '^\{.*\}$'  -- Basic JSON check for volume
    AND sp.duration ~ '^\{.*\}$'  -- Basic JSON check for duration
    AND (sp.volume::JSONB ->> 'volume_traded_bc') IS NOT NULL
    AND (sp.duration::JSONB ->> 'duration_min') IS NOT NULL
    AND (sp.volume::JSONB ->> 'volume_traded_bc')::FLOAT > 0  -- Ensure no zero values for volume_traded_bc
    AND (sp.duration::JSONB ->> 'duration_min')::FLOAT > 0  -- Ensure no zero values for duration_min
    AND (sp.basic_metrics::JSONB ->> 'total_volume_bc')::FLOAT > 0  -- Ensure no zero values for duration_min
    AND ss.base_commodity IN ('USDT', 'USD', 'TUSD', 'BUSD', 'USDC', 'BUSDT')  -- Ensure base_commodity is one of the specified values
    AND ss.source_posted_at::date >= '2024-02-16' 
    AND ss.source_posted_at::date <= '2024-10-09'
ORDER BY ss.source_posted_at DESC;