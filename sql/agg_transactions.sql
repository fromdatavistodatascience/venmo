-- calc trans made during weekday
SELECT u.user_id, COUNT (DISTINCT p.payment_id) as trans_made_week
FROM payments p
INNER JOIN users u ON u.user_id = p.actor_id
WHERE EXTRACT (DOW FROM p.date_created) NOT IN (0, 6)
GROUP BY (u.user_id)
LIMIT 5;

-- calc trans made during weekend 
SELECT u.user_id, COUNT (DISTINCT p.payment_id) as trans_made_weeknd
FROM payments p
INNER JOIN users u ON u.user_id = p.actor_id
WHERE EXTRACT (DOW FROM p.date_created) IN (0, 6)
GROUP BY (u.user_id)
LIMIT 5;

-- combine the two
SELECT wday.user_id, wday.trans_made_week, wkend.trans_made_weeknd
FROM (
        SELECT u.user_id, COUNT (DISTINCT p.payment_id) as trans_made_week
        FROM payments p
        INNER JOIN users u ON u.user_id = p.actor_id
        WHERE EXTRACT (DOW FROM p.date_created) NOT IN (0, 6)
        GROUP BY (u.user_id)
) AS wday
LEFT JOIN (
        SELECT u.user_id, COUNT (DISTINCT p.payment_id) as trans_made_weeknd
        FROM payments p
        INNER JOIN users u ON u.user_id = p.actor_id
        WHERE EXTRACT (DOW FROM p.date_created) IN (0, 6)
        GROUP BY (u.user_id)
) AS wkend
    ON wday.user_id=wkend.user_id
LIMIT 35;