-- Total number of orders shipped to each country

SELECT
	ship_country,
	COUNT(*) AS total_orders
FROM
	orders
GROUP BY
	ship_country
ORDER BY
	2 DESC;

-- Total number of orders by year

SELECT
	EXTRACT('YEAR' FROM order_date) AS year,
	COUNT(*) AS total_orders
FROM
	orders
GROUP BY
	EXTRACT('YEAR' FROM order_date)
ORDER BY
	1;

-- Total amount spent per order

SELECT
	order_id,
	SUM((unit_price * quantity) - discount)::INT AS total_amount
FROM
	order_details
GROUP BY 
	order_id
ORDER BY
	order_id;
	
-- Average number of days between shipping date and order date

SELECT
	ROUND(AVG(shipped_date - order_date), 2) AS average_shipping_delay
FROM
	orders;

-- Number of orders where shipping date is later than required date

SELECT
	COUNT(*) AS total_late_orders
FROM
	orders
WHERE
	shipped_date > required_date;
	
-- Number of products per category

SELECT 
	c.category_name,
	COUNT(p.*) AS total_products
FROM
	categories c
INNER JOIN
	products p ON c.category_id = p.category_id
GROUP BY
	c.category_name
ORDER BY
	1;

-- Products on low stock

SELECT
	product_id,
	product_name,
	units_in_stock,
	reorder_level
FROM
	products
WHERE
	units_in_stock <= reorder_level
	
-- Products that have been discontinued

SELECT
	product_id,
	product_name
FROM
	products
WHERE
	discontinued = 1;

-- Top 5 countries with the highest freights

SELECT
	ship_country,
	AVG(freight)::INT AS freight_charges
FROM
	orders
GROUP BY
	ship_country
ORDER BY
	2 DESC
LIMIT 5;

-- Customers with no orders

SELECT 
	customer_id,
	company_name,
	country
FROM
	customers
WHERE
	customer_id NOT IN
(
	SELECT 
		c.customer_id
	FROM
		customers c
	INNER JOIN orders o ON c.customer_id = o.customer_id
)

-- Top 10 customers with most orders and their total spend

SELECT
	c.company_name,
	COUNT(o.*) AS total_orders,
	SUM(d.unit_price * d.quantity - d.discount)::INT AS total_amount_spent
FROM
	customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
INNER JOIN order_details d ON o.order_id = d.order_id
GROUP BY
	c.company_name
ORDER BY
	2 DESC
LIMIT 10;

-- Customers with multiple orders within a 4-day period

WITH next_order AS
(
	SELECT
		customer_id,
		order_date,
		LEAD(order_date, 1) OVER(PARTITION BY customer_id ORDER BY order_date) AS next_order_date
	FROM 
		orders
)
SELECT
	*
FROM
	next_order
WHERE
	next_order_date - order_date <= 4;
	
-- First order from each country

SELECT
	ship_country,
	MIN(order_date) AS first_order_date
FROM
	orders
GROUP BY
	ship_country;
	
-- Top 10 selling products

SELECT
	o.product_id,
	p.product_name,
	SUM(quantity) AS total_sold
FROM
	order_details o
INNER JOIN products p ON o.product_id = p.product_id
GROUP BY
	o.product_id,
	p.product_name
ORDER BY
	3 DESC
LIMIT 10;

-- Number of employees and customers from each city

SELECT
	city,
	SUM(number_of_people) FILTER (WHERE category = 'Employee') AS number_of_employees,
	SUM(number_of_people) FILTER (WHERE category = 'Customer') AS number_of_customers
FROM
(
	SELECT
		city,
		COUNT(*) AS number_of_people,
		'Employee' AS category
	FROM
		employees
	GROUP BY
		city
	UNION ALL
	SELECT
		city,
		COUNT(*) AS number_of_people,
		'Customer' AS category
	FROM
		customers
	GROUP BY
		city
) t
GROUP BY
	city;