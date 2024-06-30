SELECT 
    sp.signal_id,
    (sp.volume::JSONB ->> 'volume_traded_bc')::FLOAT AS volume_traded_bc,
    (sp.duration::JSONB ->> 'duration_min')::FLOAT AS duration_min,
    sm.total_volume_bc,
    ss.commodity,
    ss.base_commodity,
    ss.id,
    ss.source_posted_at
FROM 
    signals_pumpsignal sp
JOIN 
    signals_presignalmetrics sm
ON 
    sp.signal_id = sm.signal_id
JOIN
    signals_signal ss
ON
    sp.id = ss.id
WHERE 
    sp.volume IS NOT NULL
    AND sp.duration IS NOT NULL
    AND sp.volume ~ '^\{.*\}$'  -- Basic JSON check for volume
    AND sp.duration ~ '^\{.*\}$'  -- Basic JSON check for duration
    AND (sp.volume::JSONB ->> 'volume_traded_bc') IS NOT NULL
    AND (sp.duration::JSONB ->> 'duration_min') IS NOT NULL
    AND (sp.volume::JSONB ->> 'volume_traded_bc')::FLOAT > 0  -- Ensure no zero values for volume_traded_bc
    AND (sp.duration::JSONB ->> 'duration_min')::FLOAT > 0  -- Ensure no zero values for duration_min
    AND sm.total_volume_bc > 0  -- Ensure no zero values for total_volume_bc
    AND ss.base_commodity IN ('USDT', 'USD', 'TUSD', 'BUSD', 'USDC', 'BUSDT');  -- Ensure base_commodity is one of the specified values
