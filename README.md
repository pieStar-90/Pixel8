# Team: Pixel8


Mashreq 🛡 Signal Dashboard
Challenge Title: AI for Social Signal Intelligence in Banking

-> Project Overview

This project is a Social Signal Dashboard built for Mashreq Bank as part of The Bounty Challenge Hackathon (Starter Pack).

The dashboard helps bank teams identify early public warning signals using synthetic data only.
It does not monitor individuals, does not use real customer data and does not trigger automated actions.

Instead, the system:
o Analyzes aggregated public chatter
o Explains why signals matter
o Shows confidence and uncertainty
o Routes insights to human reviewers for action (Human-in-the-loop)

All decisions remain human-led, supporting operational response while maintaining strong ethical boundaries.

-> Purpose

The dashboard is designed for internal Mashreq Bank teams (operations, communications, risk, and technical support) to:
o Detect early signs of service issues or rumors
o Understand public perception trends
o Review potential risks before escalation
o Respond responsibly with human judgment
o The website dashboard acts as a Human-in-the-Loop review tool, not an automated enforcement system.

-> What the System Simulates

The system simulates how people publicly discuss banking services when:
o An app or service faces issues
o Customer service experiences decline
o Brand sentiment begins to shift
o Rumors or scam-related discussions emerge
o All posts are synthetic and anonymized, generated purely for demonstration.

-> Demonstrated Scenarios

The prototype demonstrates three key scenarios relevant to Mashreq Bank:
o Service / Incident Signals
Example: Sudden spike in app login failures or service downtime

o Brand Sentiment Shifts
Example: Rising negative sentiment around customer service or response times

o Executive Insight Briefing
High-level leadership summary showing:
--Key risks
--Confidence levels
--Suggested human next steps

-> Dashboard Experience

The dashboard is designed as a scroll-based interface:

o Clicking a signal smoothly scrolls to detailed explanations
o Information is layered progressively for clarity
o Reviewers maintain full context while investigating signals

Key Sections Include:

o Displays overall risk level, trends, and top emerging signals
o Prioritizes all topics under high, medium, low and human intervention categories.
o Provides leadership with concise risk overview, confidence indicators and recommended human actions.

-> Explainable AI Logic

The system uses simple, transparent, and explainable logic to align with Responsible AI principles:

o Grouping & Classification
Synthetic posts are grouped by topic and scenario

o Trend Detection
Activity is compared across time windows to detect spikes and shifts

o Scoring
Severity Score based on volume, speed, and negativity

o  Confidence Score based on repetition and similarity

-> Uncertainty Flagging
Signals are flagged when:

o Volume is low
o Sentiment is mixed or unclear
o Every signal clearly explains why it was flagged.

-> Responsible AI & Governance

This prototype follows Responsible AI best practices:

o Synthetic data only
o No personal or customer data
o No surveillance or profiling
o No automated decision-making
o Human-in-the-loop review
o Transparent scoring and reasoning
o Clear uncertainty indicators

-> Future Enhancements

o Advanced similarity detection for rumor clustering
o Real-time simulated data feeds
o Alerting workflows for internal teams
o Enhanced audit and governance features
o Scalable deployment for enterprise environments
