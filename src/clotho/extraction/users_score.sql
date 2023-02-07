WITH avg_scores AS (
	SELECT 
		entity_id, 
		avg(channel_time_score) AS time_score, 
		avg(channel_crowd_score) AS crowd_score
	FROM cloudburst_signals
	GROUP BY entity_id
)
SELECT 
	cm."user_PID",
	sum(time_score) as time_score,
	sum(crowd_score) as crowd_score
FROM 
	telegram_chats_members cm 
	JOIN telegram_chats tc ON cm."chat_PID" = tc.pid
	JOIN avg_scores ON tc.entity_id = avg_scores.entity_id
GROUP BY 1
ORDER BY 3 DESC, 2 DESC