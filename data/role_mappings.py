"""
Role mappings and descriptions for candidate role prediction
"""

ROLE_MAPPINGS = {
    'Software Engineer': [
        'Python', 'JavaScript', 'Java', 'C++', 'C#', 'Git', 'SQL', 'HTML', 'CSS',
        'React', 'Node.js', 'Express.js', 'MongoDB', 'PostgreSQL', 'REST API',
        'Unit Testing', 'Agile', 'Problem Solving', 'Communication'
    ],
    
    'Frontend Developer': [
        'HTML', 'CSS', 'JavaScript', 'TypeScript', 'React', 'Angular', 'Vue.js',
        'Bootstrap', 'Tailwind CSS', 'SASS', 'Webpack', 'npm', 'Git',
        'Responsive Design', 'Cross-Browser Testing', 'UI/UX Design',
        'jQuery', 'REST API', 'GraphQL', 'Progressive Web Apps'
    ],
    
    'Backend Developer': [
        'Python', 'Java', 'Node.js', 'C#', 'PHP', 'Ruby', 'Go', 'SQL',
        'PostgreSQL', 'MongoDB', 'Redis', 'REST API', 'GraphQL',
        'Django', 'Flask', 'Spring Boot', 'Express.js', 'Docker',
        'AWS', 'Microservices', 'Unit Testing', 'Git', 'Linux'
    ],
    
    'Full Stack Developer': [
        'JavaScript', 'TypeScript', 'Python', 'React', 'Node.js', 'Express.js',
        'HTML', 'CSS', 'SQL', 'MongoDB', 'PostgreSQL', 'Git', 'REST API',
        'Docker', 'AWS', 'Redux', 'Next.js', 'GraphQL', 'Jest',
        'Agile', 'Problem Solving', 'Communication'
    ],
    
    'DevOps Engineer': [
        'Docker', 'Kubernetes', 'AWS', 'Azure', 'Jenkins', 'Terraform',
        'Ansible', 'Git', 'Linux', 'Python', 'Bash', 'CI/CD',
        'Monitoring', 'Prometheus', 'Grafana', 'Nginx', 'Apache',
        'Infrastructure as Code', 'Microservices', 'Networking'
    ],
    
    'Data Scientist': [
        'Python', 'R', 'SQL', 'Machine Learning', 'Pandas', 'NumPy',
        'Scikit-learn', 'TensorFlow', 'PyTorch', 'Jupyter', 'Matplotlib',
        'Seaborn', 'Statistical Analysis', 'Data Visualization',
        'Feature Engineering', 'Deep Learning', 'Natural Language Processing',
        'Time Series Analysis', 'A/B Testing', 'Big Data'
    ],
    
    'Machine Learning Engineer': [
        'Python', 'TensorFlow', 'PyTorch', 'Scikit-learn', 'Keras',
        'Machine Learning', 'Deep Learning', 'Neural Networks',
        'Model Deployment', 'MLflow', 'Docker', 'Kubernetes',
        'AWS', 'Apache Spark', 'Feature Engineering', 'Data Pipeline',
        'Git', 'Linux', 'Statistical Analysis', 'Computer Vision'
    ],
    
    'Data Engineer': [
        'Python', 'SQL', 'Apache Spark', 'Apache Kafka', 'Apache Airflow',
        'ETL', 'Data Pipeline', 'AWS', 'Google Cloud Platform',
        'PostgreSQL', 'MongoDB', 'Redis', 'Docker', 'Kubernetes',
        'Scala', 'Java', 'Hadoop', 'Hive', 'BigQuery', 'Snowflake'
    ],
    
    'Cloud Architect': [
        'AWS', 'Azure', 'Google Cloud Platform', 'Terraform', 'Kubernetes',
        'Docker', 'Microservices', 'Serverless', 'Infrastructure as Code',
        'Networking', 'Security', 'Cost Optimization', 'High Availability',
        'Disaster Recovery', 'Monitoring', 'Load Balancing', 'CDN',
        'Database Design', 'System Design', 'Architecture Patterns'
    ],
    
    'Security Engineer': [
        'Cybersecurity', 'Penetration Testing', 'Vulnerability Assessment',
        'OWASP', 'Encryption', 'PKI', 'Firewall', 'IDS/IPS', 'SIEM',
        'Incident Response', 'Malware Analysis', 'Network Security',
        'Application Security', 'Cloud Security', 'Compliance',
        'Risk Assessment', 'Python', 'Linux', 'Windows'
    ],
    
    'Mobile Developer': [
        'React Native', 'Flutter', 'Swift', 'Kotlin', 'Java', 'Objective-C',
        'iOS Development', 'Android Development', 'Xcode', 'Android Studio',
        'Mobile UI/UX', 'REST API', 'JSON', 'SQLite', 'Firebase',
        'Push Notifications', 'App Store Optimization', 'Git',
        'Mobile Testing', 'Performance Optimization'
    ],
    
    'QA Engineer': [
        'Manual Testing', 'Automation Testing', 'Selenium', 'Cypress',
        'Jest', 'JUnit', 'TestNG', 'Postman', 'API Testing',
        'Performance Testing', 'Load Testing', 'Security Testing',
        'Test Planning', 'Test Case Design', 'Bug Tracking',
        'Agile', 'Scrum', 'Regression Testing', 'User Acceptance Testing'
    ],
    
    'UI/UX Designer': [
        'User Experience Design', 'User Interface Design', 'Wireframing',
        'Prototyping', 'Figma', 'Sketch', 'Adobe XD', 'InVision',
        'User Research', 'Usability Testing', 'Information Architecture',
        'Interaction Design', 'Visual Design', 'Design Systems',
        'Responsive Design', 'Accessibility', 'HTML', 'CSS'
    ],
    
    'Product Manager': [
        'Product Strategy', 'Market Research', 'User Research',
        'Requirements Gathering', 'Roadmap Planning', 'Agile', 'Scrum',
        'Stakeholder Management', 'Data Analysis', 'A/B Testing',
        'User Stories', 'Feature Prioritization', 'Go-to-Market Strategy',
        'Competitive Analysis', 'KPI Tracking', 'Communication'
    ],
    
    'Project Manager': [
        'Project Management', 'Agile', 'Scrum', 'Kanban', 'PMP',
        'Risk Management', 'Budget Management', 'Timeline Management',
        'Stakeholder Communication', 'Team Leadership', 'Resource Planning',
        'Quality Assurance', 'Change Management', 'Problem Solving',
        'JIRA', 'Confluence', 'MS Project', 'Gantt Charts'
    ],
    
    'Business Analyst': [
        'Requirements Analysis', 'Business Process Modeling',
        'Data Analysis', 'SQL', 'Excel', 'Power BI', 'Tableau',
        'Stakeholder Management', 'Documentation', 'Process Improvement',
        'Gap Analysis', 'Use Case Development', 'Wireframing',
        'Agile', 'Scrum', 'Communication', 'Problem Solving'
    ],
    
    'System Administrator': [
        'Linux', 'Windows Server', 'Networking', 'Active Directory',
        'DNS', 'DHCP', 'Firewall', 'VPN', 'Backup and Recovery',
        'Monitoring', 'Virtualization', 'VMware', 'Hyper-V',
        'Shell Scripting', 'PowerShell', 'Python', 'Security',
        'Hardware Troubleshooting', 'Documentation'
    ],
    
    'Database Administrator': [
        'SQL', 'MySQL', 'PostgreSQL', 'Oracle', 'SQL Server',
        'Database Design', 'Performance Tuning', 'Backup and Recovery',
        'Replication', 'High Availability', 'Security', 'Monitoring',
        'Data Migration', 'Capacity Planning', 'Indexing',
        'Query Optimization', 'Shell Scripting', 'Python'
    ],
    
    'Technical Writer': [
        'Technical Writing', 'Documentation', 'API Documentation',
        'User Manuals', 'Help Systems', 'Content Management',
        'Markdown', 'HTML', 'CSS', 'Version Control', 'Git',
        'Agile', 'Scrum', 'Communication', 'Research Skills',
        'Information Architecture', 'Style Guides', 'Editing'
    ],
    
    'Sales Engineer': [
        'Technical Sales', 'Solution Architecture', 'Product Demos',
        'Customer Presentations', 'Requirements Gathering',
        'Proposal Writing', 'CRM', 'Salesforce', 'Communication',
        'Negotiation', 'Technical Knowledge', 'Industry Knowledge',
        'Relationship Building', 'Problem Solving', 'Travel'
    ]
}

ROLE_DESCRIPTIONS = {
    'Software Engineer': 'Designs, develops, and maintains software applications using various programming languages and technologies. Works on both frontend and backend systems.',
    
    'Frontend Developer': 'Specializes in creating user interfaces and user experiences for web applications. Focuses on client-side technologies and responsive design.',
    
    'Backend Developer': 'Develops server-side logic, databases, and APIs. Ensures the behind-the-scenes functionality of web applications works smoothly.',
    
    'Full Stack Developer': 'Works on both frontend and backend development. Has comprehensive knowledge of the entire web development stack.',
    
    'DevOps Engineer': 'Bridges the gap between development and operations. Focuses on automation, CI/CD, infrastructure, and deployment processes.',
    
    'Data Scientist': 'Analyzes complex data to extract insights and build predictive models. Uses statistical methods and machine learning techniques.',
    
    'Machine Learning Engineer': 'Develops and deploys machine learning models into production systems. Combines software engineering with ML expertise.',
    
    'Data Engineer': 'Builds and maintains data infrastructure, pipelines, and systems. Ensures data is available, reliable, and accessible for analysis.',
    
    'Cloud Architect': 'Designs and oversees cloud computing strategies and infrastructure. Ensures scalable, secure, and cost-effective cloud solutions.',
    
    'Security Engineer': 'Protects systems and networks from cyber threats. Implements security measures and responds to security incidents.',
    
    'Mobile Developer': 'Creates applications for mobile devices (iOS/Android). Focuses on mobile-specific technologies and user experiences.',
    
    'QA Engineer': 'Ensures software quality through testing and validation. Develops test strategies and automated testing frameworks.',
    
    'UI/UX Designer': 'Designs user interfaces and experiences. Focuses on usability, accessibility, and visual design of digital products.',
    
    'Product Manager': 'Manages the development and strategy of products. Works with cross-functional teams to deliver value to users.',
    
    'Project Manager': 'Oversees project execution from planning to completion. Manages timelines, resources, and stakeholder communication.',
    
    'Business Analyst': 'Analyzes business processes and requirements. Bridges the gap between business needs and technical solutions.',
    
    'System Administrator': 'Manages and maintains computer systems and networks. Ensures system reliability, security, and performance.',
    
    'Database Administrator': 'Manages database systems and ensures data integrity, security, and performance. Handles backup and recovery processes.',
    
    'Technical Writer': 'Creates technical documentation, user guides, and help systems. Translates complex technical information into clear content.',
    
    'Sales Engineer': 'Combines technical expertise with sales skills. Helps customers understand and implement technical solutions.'
}
