MITRE_ATTACK_DB = {

    "BENIGN": {
        "technique": "None",
        "technique_id": "N/A",
        "tactic": "None",
        "recommendations": [
            "No action required."
        ]
    },
    "Benign": {
        "technique": "None",
        "technique_id": "N/A",
        "tactic": "None",
        "recommendations": [
            "No action required."
        ]
    },

    "Bot Attack": {
        "technique": "Application Layer Protocol",
        "technique_id": "T1071",
        "tactic": "Command and Control",
        "recommendations": [
            "Block malicious IP addresses",
            "Enable bot detection mechanisms",
            "Monitor outbound communications",
            "Inspect DNS traffic"
        ]
    },

    "DDoS": {
        "technique": "Network Denial of Service",
        "technique_id": "T1498",
        "tactic": "Impact",
        "recommendations": [
            "Enable rate limiting",
            "Deploy DDoS mitigation services",
            "Filter malicious traffic",
            "Monitor bandwidth utilization"
        ]
    },

    "DoS GoldenEye": {
        "technique": "Endpoint Denial of Service",
        "technique_id": "T1499",
        "tactic": "Impact",
        "recommendations": [
            "Deploy Web Application Firewall",
            "Monitor HTTP traffic spikes",
            "Limit concurrent requests"
        ]
    },

    "DoS Hulk": {
        "technique": "Endpoint Denial of Service",
        "technique_id": "T1499",
        "tactic": "Impact",
        "recommendations": [
            "Apply traffic filtering",
            "Enable request throttling",
            "Monitor web server resources"
        ]
    },

    "DoS Slowhttptest": {
        "technique": "Slow HTTP Attack",
        "technique_id": "T1499",
        "tactic": "Impact",
        "recommendations": [
            "Reduce timeout settings",
            "Deploy reverse proxies",
            "Limit idle connections"
        ]
    },

    "DoS slowloris": {
        "technique": "Slow HTTP Attack",
        "technique_id": "T1499",
        "tactic": "Impact",
        "recommendations": [
            "Configure connection limits",
            "Use reverse proxy servers",
            "Monitor half-open connections"
        ]
    },

    "FTP-Patator": {
        "technique": "Brute Force",
        "technique_id": "T1110",
        "tactic": "Credential Access",
        "recommendations": [
            "Enable account lockout",
            "Use multi-factor authentication",
            "Monitor failed login attempts",
            "Enforce strong password policies"
        ]
    },

    "Brute Force": {
        "technique": "Brute Force",
        "technique_id": "T1110",
        "tactic": "Credential Access",
        "recommendations": [
            "Disable password authentication",
            "Use SSH keys",
            "Enable MFA",
            "Restrict SSH access"
        ]
    },

    "Infiltration": {
        "technique": "Ingress Tool Transfer",
        "technique_id": "T1105",
        "tactic": "Command and Control",
        "recommendations": [
            "Isolate affected host immediately",
            "Perform forensic analysis",
            "Scan the network for compromise",
            "Rotate credentials immediately"
        ]
    },

    "Port Scan": {
        "technique": "Network Service Discovery",
        "technique_id": "T1046",
        "tactic": "Discovery",
        "recommendations": [
            "Close unused ports",
            "Restrict external access",
            "Monitor scanning attempts",
            "Enable firewall logging"
        ]
    },

    "Web Attack - Brute Force": {
        "technique": "Brute Force",
        "technique_id": "T1110",
        "tactic": "Credential Access",
        "recommendations": [
            "Enable account lockout",
            "Deploy CAPTCHA",
            "Monitor authentication logs"
        ]
    },

    "Web Attack - Sql Injection": {
        "technique": "Exploit Public Facing Application",
        "technique_id": "T1190",
        "tactic": "Initial Access",
        "recommendations": [
            "Use parameterized SQL queries",
            "Validate user input",
            "Deploy a WAF",
            "Perform secure code review"
        ]
    },

    "Web Attack - XSS": {
        "technique": "Exploit Public Facing Application",
        "technique_id": "T1190",
        "tactic": "Initial Access",
        "recommendations": [
            "Sanitize user input",
            "Enable Content Security Policy",
            "Encode outputs properly",
            "Validate all form inputs"
        ]
    }
}