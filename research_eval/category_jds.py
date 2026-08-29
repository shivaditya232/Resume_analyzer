"""
One job description per resume category used from the Kaggle 'Resume dataset.csv'
(haidermaseeh/resume-dataset — 9,000 real resumes). Originally 9 CSV categories,
but "Recruiter Resumes" was dropped (see note below) leaving 8.

Each JD is paired against real resumes selected by CONTENT (job_title match, see
large_test_dataset.py) rather than trusting the CSV's own 'category' column --
that column was found to be unreliable for several categories (e.g. most
"Web Developer Resumes" rows are actually ETL/Data-Warehousing resumes).
Keys here are still named after the original CSV category labels for continuity/
readability, but large_test_dataset.py no longer looks resumes up by that column.
"""

CATEGORY_JDS = {
    "Java Developers/Architects Resumes": """
Senior Java Developer / Architect

We are hiring a Java Developer / Architect to design and build enterprise-grade
backend systems.

Responsibilities:
- Design and develop applications using Core Java, J2EE/JEE (Servlets, JSP, EJB, JMS, JDBC).
- Build and consume Web Services (SOAP/REST, JAX-WS, JAX-RS).
- Apply OOAD principles; produce UML design artifacts (class/sequence diagrams).
- Participate in full SDLC: requirements, design, coding, testing, deployment.
- Work with relational databases (Oracle, SQL Server) and ORM frameworks (Hibernate, JPA).

Requirements:
- 5+ years professional Java/J2EE development experience.
- Strong knowledge of design patterns, MVC architecture, and multi-tier systems.
- Experience with application servers (WebLogic, WebSphere, JBoss) and build tools (Maven/Ant).
- Bachelor's degree in Computer Science or related field.
- Nice to have: Spring, Hibernate, microservices experience.
""",

    "Web Developer Resumes": """
Web Developer

We're looking for a Web Developer to build and maintain responsive, client-facing
web applications.

Responsibilities:
- Develop front-end interfaces using HTML5, CSS3, JavaScript, and modern frameworks
  (React, Angular, or similar).
- Build and integrate back-end services and APIs (Node.js, PHP, ASP.NET, or Java).
- Ensure cross-browser compatibility and responsive design across devices.
- Collaborate with designers and back-end teams to deliver full web solutions.
- Optimize applications for speed, scalability, and SEO.

Requirements:
- 2-5 years of web development experience (front-end and/or full-stack).
- Proficiency in JavaScript, HTML/CSS, and at least one modern JS framework.
- Familiarity with REST APIs, version control (Git), and basic database use (SQL/NoSQL).
- Bachelor's degree in Computer Science or related field preferred.
- Nice to have: TypeScript, cloud deployment experience (AWS/Azure).
""",

    "SQL Developers Resumes": """
SQL Developer / Database Developer

We are seeking a SQL Developer to design, develop, and optimize database solutions.

Responsibilities:
- Write and optimize complex T-SQL/PL-SQL queries, stored procedures, functions, and triggers.
- Design normalized database schemas and maintain data integrity.
- Perform query performance tuning, indexing, and execution plan analysis.
- Develop ETL scripts and support data migration/integration efforts.
- Collaborate with application developers to support database-driven applications.

Requirements:
- 3+ years of hands-on experience with SQL Server, Oracle, or MySQL.
- Strong skills in T-SQL/PL-SQL, stored procedures, and query optimization.
- Experience with database design, normalization, and indexing strategies.
- Familiarity with SSIS/SSRS or equivalent ETL/reporting tools is a plus.
- Bachelor's degree in Computer Science, Information Systems, or related field.
""",

    "Business Analyst (BA) Resumes": """
Business Analyst

We are looking for a Business Analyst to bridge business needs and technical solutions.

Responsibilities:
- Elicit, document, and validate business and functional requirements from stakeholders.
- Create use cases, process flow diagrams, and requirement specification documents (BRD/FRD).
- Facilitate JAD sessions, stakeholder interviews, and requirement walkthroughs.
- Work with QA teams to define test plans and validate that solutions meet requirements.
- Support UAT and assist in change management/process improvement initiatives.

Requirements:
- 3+ years of experience as a Business Analyst, preferably in a corporate/enterprise setting.
- Strong skills in requirements gathering, gap analysis, and stakeholder management.
- Experience with tools like JIRA, Confluence, Visio, or similar.
- Familiarity with Agile/Scrum and Waterfall methodologies.
- Bachelor's degree in Business, Information Systems, or related field.
""",

    "Network and Systems Administrators Resumes": """
Network and Systems Administrator

We need a Network/Systems Administrator to manage and maintain our IT infrastructure.

Responsibilities:
- Configure, maintain, and troubleshoot LAN/WAN, routers, switches, and firewalls.
- Administer Windows Server/Linux systems, Active Directory, and DNS/DHCP services.
- Monitor network performance and security; respond to incidents and outages.
- Manage backups, disaster recovery, and system patching/updates.
- Support virtualization environments (VMware/Hyper-V) and cloud infrastructure.

Requirements:
- 3+ years of experience in network and/or systems administration.
- Strong knowledge of TCP/IP, VPN, firewalls, and network security practices.
- Hands-on experience with Windows Server and/or Linux administration.
- Certifications such as CCNA, CompTIA Network+/Security+, or MCSE are a plus.
- Bachelor's degree in IT, Computer Science, or related field preferred.
""",

    "Datawarehousing, ETL, Informatica Resumes": """
ETL / Data Warehouse Developer (Informatica)

We are hiring an ETL Developer to build and maintain data warehousing pipelines.

Responsibilities:
- Design, develop, and maintain ETL workflows using Informatica PowerCenter or similar tools.
- Build and optimize data warehouse schemas (star/snowflake schema, fact and dimension tables).
- Perform data extraction, transformation, and loading from multiple heterogeneous sources.
- Conduct data quality checks, validation, and reconciliation between source and target systems.
- Collaborate with BI teams to support reporting and analytics needs.

Requirements:
- 3+ years of experience with ETL development using Informatica, SSIS, or DataStage.
- Strong SQL skills and understanding of data warehousing concepts (OLAP, dimensional modeling).
- Experience with relational databases (Oracle, SQL Server, Teradata).
- Familiarity with job scheduling tools and performance tuning of ETL jobs.
- Bachelor's degree in Computer Science, Information Systems, or related field.
""",

    "Business Intelligence, Business Object Resumes": """
Business Intelligence Developer (Enterprise Reporting & Analytics)

We are looking for a BI Developer to design reporting and analytics solutions on
whichever enterprise BI platform the candidate specializes in -- SAP Business
Objects (Web Intelligence, Crystal Reports, Universe Designer), Oracle OBIEE,
Power BI, Tableau, or Cognos are all relevant background.

Responsibilities:
- Design and develop reports, dashboards, and semantic/universe layers using an
  enterprise BI platform (e.g. SAP Business Objects, OBIEE, Power BI, Tableau, or Cognos).
- Translate business requirements into effective BI reports and visualizations.
- Optimize report performance and manage repository/security administration for
  the BI platform in use.
- Work with data warehouse teams to ensure accurate, timely data for reporting.
- Support ad-hoc reporting requests and train end users on self-service BI tools.

Requirements:
- 3+ years of experience with an enterprise BI/reporting platform (SAP Business
  Objects, OBIEE, Power BI, Tableau, or Cognos).
- Strong SQL skills and understanding of data warehousing/dimensional modeling.
- Experience with report performance tuning and BI platform administration.
- Familiarity with additional BI tools beyond your primary platform is a plus.
- Bachelor's degree in Computer Science, Information Systems, or related field.
""",

    "Project Manager Resumes": """
IT Project Manager

We are seeking a Project Manager to lead cross-functional technology projects
from initiation to delivery.

Responsibilities:
- Develop and manage project plans, schedules, budgets, and resource allocation.
- Coordinate cross-functional teams (development, QA, business stakeholders).
- Identify risks, manage scope changes, and resolve project roadblocks.
- Track project status and report progress to leadership and stakeholders.
- Ensure projects are delivered on time, within budget, and meet quality standards.

Requirements:
- 5+ years of experience managing IT/software projects.
- Strong knowledge of project management methodologies (Agile, Scrum, Waterfall).
- PMP or Scrum Master certification preferred.
- Excellent communication, stakeholder management, and leadership skills.
- Bachelor's degree in Business, Computer Science, or related field.
""",

    # NOTE: "Recruiter Resumes" was dropped. Verified three separate ways (CSV
    # category label, full-text keyword scan, and job_title field scan) that this
    # Kaggle dataset contains essentially no genuine recruiter resumes (6 out of
    # 9,000 total by job_title). Rather than force an unreliable tiny sample, the
    # study uses these 8 categories.
}
