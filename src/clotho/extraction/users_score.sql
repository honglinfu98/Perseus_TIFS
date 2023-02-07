CREATE TEMPORARY TABLE temp_table as
	SELECT avg(channel_time_score) as time_score, avg(channel_crowd_score) as crowd_score, entity_id
	FROM cloudburst_signals
	GROUP BY entity_id;

CREATE TEMPORARY TABLE temp_table2 as 
	SELECT 
	cm."user_PID",
	tc.entity_id
	FROM
	public.telegram_chats_members cm LEFT JOIN public.telegram_chats tc on cm."chat_PID" = tc.pid;
	

SELECT 
cm."user_PID",
sum(time_score),
sum(crowd_score)
FROM 
(public.telegram_chats_members cm LEFT JOIN public.telegram_chats tc on cm."chat_PID" = tc.pid)
LEFT JOIN temp_table on tc.entity_id = temp_table.entity_id
GROUP BY 1
ORDER BY 3,2