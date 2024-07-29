-- Highest, lowest, total and average salaries of employees by department

SELECT
	d.department_name,
	ROUND(MAX(e.salary), 2) AS highest_salary,
	ROUND(MIN(e.salary), 2) AS lowest_salary,
	ROUND(SUM(e.salary), 2) AS total_salary,
	ROUND(AVG(e.salary), 2) AS average_salary
FROM
	employees e
INNER JOIN departments d ON e.department_id = d.department_id
GROUP BY
	d.department_name;

-- Difference between lowest and highest salaries per job

SELECT
	job_title,
	max_salary - min_salary AS gap_in_salaries
FROM
	jobs;
	
-- Average salary for each department with more than 10 employees

SELECT
	d.department_name,
	ROUND(AVG(e.salary), 2) AS average_salary,
	COUNT(*) AS number_of_employees
FROM
	employees e
INNER JOIN departments d ON e.department_id = d.department_id
GROUP BY
	d.department_name
HAVING
	COUNT(*) > 10;

-- Average salary for each post excluding programmer

SELECT
	j.job_title,
	ROUND(AVG(e.salary), 2) AS average_salary
FROM
	jobs j
INNER JOIN
	employees e ON j.job_id = e.job_id
WHERE
	j.job_id <> 'IT_PROG'
GROUP BY
	j.job_title;

-- Top 10 highest paid employees

SELECT
	e.first_name,
	e.last_name,
	j.job_title,
	e.salary
FROM
	employees e
INNER JOIN 
	jobs j ON e.job_id = j.job_id
ORDER BY
	e.salary DESC
LIMIT 10;

-- All employees whose first name starts with A, C or M

SELECT
	first_name
FROM
	employees
WHERE
	SUBSTRING(first_name, 1, 1) IN ('A', 'C', 'M');

-- All employees with the letter C in their last name at the third or greater position;

SELECT
	last_name
FROM
	employees
WHERE
	POSITION('c' IN last_name) >= 3;

-- Replace all phone_numbers with '123' in them with the substring '888'

UPDATE employees
SET phone_number = REPLACE(phone_number, '123', '888');

-- Top 10 longest serving employees

SELECT
	e.first_name,
	e.last_name,
	e.hire_date
FROM
	employees e
ORDER BY
	3
LIMIT 10;

-- Number of employees per country

SELECT
	c.country_name,
	COUNT(*) AS number_of_employees
FROM
	countries c
INNER JOIN locations l ON c.country_id = l.country_id
INNER JOIN departments d ON l.location_id = d.location_id
INNER JOIN employees e ON d.department_id = e.department_id
GROUP BY
	c.country_name;
	
-- The commission and total salary of each employee

SELECT
	first_name,
	last_name,
	salary,
	commission_pct * salary AS commission,
	salary + commission_pct * salary AS total_salary
FROM
	employees
