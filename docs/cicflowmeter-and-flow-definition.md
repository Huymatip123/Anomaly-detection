# CICFlowMeter & Network Flow Definition

A deep dive into how raw network packets become the 84 statistical features our model uses.

---

## Table of Contents

1. [What Is a Network Flow?](#1-what-is-a-network-flow)
2. [The 5-Tuple: The Identity of a Flow](#2-the-5-tuple-the-identity-of-a-flow)
3. [How CICFlowMeter Builds Flows from Packets](#3-how-cicflowmeter-builds-flows-from-packets)
4. [Forward vs Backward Direction](#4-forward-vs-backward-direction)
5. [The 84 Statistical Features](#5-the-84-statistical-features)
   - 5.1 [Flow Identification (6 columns)](#51-flow-identification-6-columns)
   - 5.2 [Flow Timestamps & Duration](#52-flow-timestamps--duration)
   - 5.3 [Packet Length Statistics](#53-packet-length-statistics)
   - 5.4 [Packet Counts & Rates](#54-packet-counts--rates)
   - 5.5 [Inter-Arrival Time (IAT) Statistics](#55-inter-arrival-time-iat-statistics)
   - 5.6 [TCP Flags](#56-tcp-flags)
   - 5.7 [Window & Segment Size](#57-window--segment-size)
   - 5.8 [Active / Idle Statistics](#58-active--idle-statistics)
   - 5.9 [Subflow Statistics](#59-subflow-statistics)
   - 5.10 [Derived Ratios](#510-derived-ratios)
6. [Example: One Flow from Start to Finish](#6-example-one-flow-from-start-to-finish)
7. [How This Connects to Our Dataset](#7-how-this-connects-to-our-dataset)
8. [How This Connects to Real-Time Deployment](#8-how-this-connects-to-real-time-deployment)

---

## 1. What Is a Network Flow?

A **network flow** (also called a "bidirectional flow" or "bi-flow") is the complete record of a conversation between two computers on a network.

Imagine you (Computer A) visit a website (Computer B):

```
You:  "Hey, I want to open a connection"          [SYN]
Server: "OK, connection opened"                    [SYN-ACK]
You:  "Great, now send me the homepage"            [ACK + GET /]
Server: "Here's the HTML, images, CSS..."           [Data packets]
You:  "Got it, goodbye"                             [FIN]
Server: "Goodbye"                                   [FIN-ACK]
```

A flow records **both directions** of this entire conversation as one unit. It does NOT record the actual content (the HTML page). It only records **statistics** about how the conversation happened.

---

## 2. The 5-Tuple: The Identity of a Flow

Every flow is uniquely identified by **five fields** — the 5-tuple:

| # | Field | Example | Purpose |
|---|-------|---------|---------|
| 1 | **Source IP** | `192.168.1.10` | Who started the conversation |
| 2 | **Source Port** | `54321` | Which program on the source |
| 3 | **Destination IP** | `10.0.0.4` | Who was contacted |
| 4 | **Destination Port** | `80` | Which service on the destination |
| 5 | **Protocol** | `6 (TCP)` | The communication protocol |

**Key rule**: If ANY of these five fields differ, it is a different flow.

Examples:

| Src IP | Src Port | Dst IP | Dst Port | Proto | Same Flow? |
|--------|----------|--------|----------|-------|------------|
| A | 12345 | B | 80 | TCP | — (original) |
| A | 12345 | B | 80 | TCP | ✅ Same |
| A | **12346** | B | 80 | TCP | ❌ Different (different port) |
| A | 12345 | B | **443** | TCP | ❌ Different (different service) |
| A | 12345 | **C** | 80 | TCP | ❌ Different (different server) |
| **D** | 12345 | B | 80 | TCP | ❌ Different (different client) |
| A | 12345 | B | 80 | **UDP** | ❌ Different (different protocol) |

### Why the 5-Tuple Matters for IDS

An attacker typically uses one source IP to send many requests. The 5-tuple allows us to:

- **Group** all packets from the same session
- **Count** flows per source IP (detect scanning behavior)
- **Identify** services being targeted (destination port 22 = SSH brute force)
- **Distinguish** normal web traffic (port 80/443) from database attacks (port 3306)

---

## 3. How CICFlowMeter Builds Flows from Packets

CICFlowMeter is a tool that reads a `.pcap` file (raw packet capture) and produces a CSV file where each row is one flow.

### Step-by-Step Process

```
Raw Packets (PCAP)
│
│  Packet 1: A:54321 → B:80   [SYN]      Timestamp: 0.000s
│  Packet 2: B:80 → A:54321   [SYN-ACK]  Timestamp: 0.001s
│  Packet 3: A:54321 → B:80   [ACK]      Timestamp: 0.002s
│  Packet 4: A:54321 → B:80   [GET /]    Timestamp: 0.003s
│  Packet 5: B:80 → A:54321   [DATA 1460]Timestamp: 0.010s
│  ...
│  Packet N: A:54321 → B:80   [FIN]      Timestamp: 1.200s
│
▼
┌──────────────────────────────────────────────────────────┐
│ Step 1: Extract 5-tuple from each packet                │
│   (src_ip, src_port, dst_ip, dst_port, protocol)        │
└──────────────────────────────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────┐
│ Step 2: Group packets by 5-tuple                        │
│   All packets with same 5-tuple → one flow              │
│   Packets are sorted by timestamp within each flow      │
└──────────────────────────────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────┐
│ Step 3: Determine forward/backward direction            │
│   First packet's direction = forward (src→dst)          │
│   Opposite direction = backward (dst→src)               │
└──────────────────────────────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────┐
│ Step 4: Compute 84 statistical features                 │
│   Duration, packet counts, byte counts, IAT,            │
│   TCP flags, window sizes, etc.                         │
│   (Each statistic is calculated SEPARATELY for          │
│    forward and backward directions)                     │
└──────────────────────────────────────────────────────────┘
│
▼
Output CSV: one row = one flow

FlowID, SrcIP, SrcPort, DstIP, DstPort, Protocol,
Flow Duration, Tot Fwd Pkts, Tot Bwd Pkts,
Fwd Pkt Len Mean, Bwd Pkt Len Mean,
Fwd IAT Mean, Bwd IAT Mean,
... (80+ more features)
```

### Flow Termination Conditions

When does a flow end?

| Condition | TCP | UDP |
|-----------|-----|-----|
| **Normal teardown** | FIN or RST packet | Never (UDP has no teardown) |
| **Timeout** | Flow timeout (default: 600s) | Flow timeout (default: 600s) |
| **Idle timeout** | No packets for N seconds | No packets for N seconds |

If a flow is very long (e.g., a video stream), CICFlowMeter splits it into multiple flow records at the timeout boundary.

---

## 4. Forward vs Backward Direction

This is one of the most important concepts in CICFlowMeter.

```
First packet: A → B  (Forward direction defined)

All future packets:
  A → B  =  Forward (Fwd)
  B → A  =  Backward (Bwd)
```

**Why separate directions matter for intrusion detection:**

| Attack | Forward Pattern | Backward Pattern |
|--------|----------------|------------------|
| **DoS flood** | Many small packets, zero/very few bytes backward | Almost no backward traffic |
| **Data exfiltration** | Few requests, very large responses | Bwd bytes >> Fwd bytes |
| **Port scan** | Many SYN packets, no completion | No backward or RST |
| **Normal web** | Small request, large response | Bwd bytes > Fwd bytes (but balanced) |
| **DNS tunneling** | Many small queries, slightly larger responses | Unusual packet size ratio |

Every feature in CICFlowMeter comes in forward/backward pairs:

| Feature | Forward | Backward |
|---------|---------|----------|
| Packet count | `Tot Fwd Pkts` | `Tot Bwd Pkts` |
| Packet length max | `Fwd Pkt Len Max` | `Bwd Pkt Len Max` |
| Packet length mean | `Fwd Pkt Len Mean` | `Bwd Pkt Len Mean` |
| Packet length std | `Fwd Pkt Len Std` | `Bwd Pkt Len Std` |
| Inter-arrival time mean | `Fwd IAT Mean` | `Bwd IAT Mean` |
| Inter-arrival time std | `Fwd IAT Std` | `Bwd IAT Std` |
| Packets per second | `Fwd Pkts/s` | `Bwd Pkts/s` |
| Segment size | `Fwd Seg Size Avg` | `Bwd Seg Size Avg` |

---

## 5. The 84 Statistical Features

A complete breakdown of every feature group in CICFlowMeter. We use the feature names as they appear in the CSV.

### 5.1 Flow Identification (6 columns)

These identify the flow. They are used during flow extraction but often **removed** from the training data (to prevent the model from learning IP-based patterns instead of behavior-based patterns).

| Column | Example | Description |
|--------|---------|-------------|
| `Flow ID` | `192.168.1.10-54321-10.0.0.4-80-6` | Concatenation of 5-tuple |
| `Src IP` | `192.168.1.10` | Source IP address |
| `Src Port` | `54321` | Source port number |
| `Dst IP` | `10.0.0.4` | Destination IP address |
| `Dst Port` | `80` | Destination port number |
| `Protocol` | `6` | IP protocol (6=TCP, 17=UDP) |

**Note:** In the CIC-UNSW-NB15 dataset (Data.csv), IP addresses are removed for privacy. Only `Dst Port` and `Protocol` remain.

### 5.2 Flow Timestamps & Duration

| Column | Unit | Description |
|--------|------|-------------|
| `Flow Duration` | Microseconds (μs) | Time between first and last packet in the flow |

Duration is critical for classification:

| Attack | Typical Duration |
|--------|-----------------|
| DoS flood | Very short (< 1ms per flow) |
| Port scan | Very short (single SYN) |
| Normal web browsing | Medium (100ms - 10s) |
| Data exfiltration | Long (minutes) |
| Worms propagation | Variable, but many short flows |

### 5.3 Packet Length Statistics

Each statistic is computed separately for forward and backward directions.

| Column | Description | Attack Relevance |
|--------|-------------|------------------|
| `Fwd Pkt Len Max` | Largest forward packet | DoS: very small packets |
| `Fwd Pkt Len Min` | Smallest forward packet | Normal: includes ACK-only (40 bytes) |
| `Fwd Pkt Len Mean` | Average forward packet size | Data exfil: unusually large |
| `Fwd Pkt Len Std` | Std deviation of fwd packet sizes | Fuzzing: high variance |
| `Bwd Pkt Len Max` | Largest backward packet | Normal: ~1460 (MSS) |
| `Bwd Pkt Len Min` | Smallest backward packet | |
| `Bwd Pkt Len Mean` | Average backward packet size | Shellcode: small responses |
| `Bwd Pkt Len Std` | Std deviation of bwd sizes | |
| `Pkt Len Max` | Overall max (any direction) | |
| `Pkt Len Min` | Overall min (any direction) | |
| `Pkt Len Mean` | Overall mean | |
| `Pkt Len Std` | Overall std deviation | |
| `Pkt Len Var` | Overall variance | |

**Example interpretation:**

```
Normal web:       Fwd Pkt Len Mean ≈ 300  (HTTP request)
                  Bwd Pkt Len Mean ≈ 1200 (HTTP response with data)
                  
DoS flood:        Fwd Pkt Len Mean ≈ 40   (SYN-only, no data)
                  Bwd Pkt Len Mean ≈ 0    (no response)

Data exfiltration: Fwd Pkt Len Mean ≈ 1400 (large outgoing data)
                   Bwd Pkt Len Mean ≈ 40   (tiny ACK responses)
```

### 5.4 Packet Counts & Rates

| Column | Unit | Description |
|--------|------|-------------|
| `Tot Fwd Pkts` | Count | Total forward packets |
| `Tot Bwd Pkts` | Count | Total backward packets |
| `TotLen Fwd Pkts` | Bytes | Total bytes in forward direction |
| `TotLen Bwd Pkts` | Bytes | Total bytes in backward direction |
| `Fwd Pkts/s` | Packets/sec | Forward packet rate |
| `Bwd Pkts/s` | Packets/sec | Backward packet rate |
| `Down/Up Ratio` | Ratio | Bwd packets / Fwd packets |

**Key relationships:**

- `TotLen Fwd Pkts / Tot Fwd Pkts ≈ Fwd Pkt Len Mean`
- `Fwd Pkts/s = Tot Fwd Pkts / (Flow Duration / 1,000,000)`
- `Down/Up Ratio > 10` → likely a request-response protocol (like HTTP)

### 5.5 Inter-Arrival Time (IAT) Statistics

IAT = time between consecutive packets in the same direction.

| Column | Unit | Description |
|--------|------|-------------|
| `Fwd IAT Max` | μs | Max forward inter-arrival time |
| `Fwd IAT Min` | μs | Min forward inter-arrival time |
| `Fwd IAT Mean` | μs | Average forward inter-arrival time |
| `Fwd IAT Std` | μs | Std deviation of forward IAT |
| `Fwd IAT Tot` | μs | Sum of all forward IATs |
| `Bwd IAT Max` | μs | Max backward inter-arrival time |
| `Bwd IAT Min` | μs | Min backward inter-arrival time |
| `Bwd IAT Mean` | μs | Average backward inter-arrival time |
| `Bwd IAT Std` | μs | Std deviation of backward IAT |
| `Bwd IAT Tot` | μs | Sum of all backward IATs |
| `Flow IAT Max` | μs | Max IAT across ALL packets in flow |
| `Flow IAT Min` | μs | Min IAT across all packets |
| `Flow IAT Mean` | μs | Mean IAT across all packets |
| `Flow IAT Std` | μs | Std deviation of all IATs |

**Why IAT matters for intrusion detection:**

| Pattern | IAT Signature |
|---------|---------------|
| **Human browsing** | Irregular IAT: long pauses (reading), then bursts (page loads) |
| **Automated tool / bot** | Very regular IAT: constant intervals between requests |
| **DoS flood** | Extremely low IAT: packets arrive at line rate |
| **Slowloris attack** | Deliberately high IAT: sends headers slowly to keep connection open |
| **Port scan** | Very low IAT with high variation (scanning many ports quickly) |

### 5.6 TCP Flags

Each flag column counts how many packets in the flow had that flag set.

| Column | Flag | Description | Attack Relevance |
|--------|------|-------------|------------------|
| `FIN Flag Cnt` | FIN | End of connection | Normal in clean teardown |
| `SYN Flag Cnt` | SYN | Connection request | Port scan: many SYN without completion |
| `RST Flag Cnt` | RST | Connection reset | Scanning: port closed → RST |
| `PSH Flag Cnt` | PSH | Push data immediately | Data exfil: urgent pushing |
| `ACK Flag Cnt` | ACK | Acknowledgment | Normal traffic |
| `URG Flag Cnt` | URG | Urgent pointer | Rare in normal traffic |
| `CWE Flag Count` | CWE | Congestion Window Reduced | Rare |
| `ECE Flag Cnt` | ECE | ECN Echo | Rare |

**Attack signatures from flags:**

- **SYN flood**: High `SYN Flag Cnt`, low `ACK Flag Cnt`, high `Fwd Pkts/s`
- **Port scan**: High `SYN Flag Cnt`, high `RST Flag Cnt` (closed ports → RST)
- **Normal HTTP**: Balanced SYN/ACK, one or zero FIN, variable PSH
- **Stealth scan**: SYN packets only, no completion

### 5.7 Window & Segment Size

| Column | Description |
|--------|-------------|
| `Init Fwd Win Byts` | Initial forward window size (advertised by receiver) |
| `Init Bwd Win Byts` | Initial backward window size |
| `Fwd Seg Size Avg` | Average forward segment size (payload only) |
| `Bwd Seg Size Avg` | Average backward segment size |
| `Fwd Seg Size Min` | Min forward segment size |

Window size tells us about the **operating system** on the other end:

| OS | Initial Window Size |
|----|-------------------|
| Linux | 29200 or 5840 |
| Windows 10 | 65535 |
| macOS | 65535 |
| FreeBSD | 65535 |

Attackers using custom tools often have non-standard window sizes — this can be a detection signal.

### 5.8 Active / Idle Statistics

These describe sub-activity within a flow. "Active" means packets are flowing. "Idle" means a gap with no packets.

| Column | Unit | Description |
|--------|------|-------------|
| `Active Mean` | μs | Average duration of active bursts |
| `Active Std` | μs | Std deviation of active burst durations |
| `Active Max` | μs | Maximum active burst duration |
| `Active Min` | μs | Minimum active burst duration |
| `Idle Mean` | μs | Average idle period duration |
| `Idle Std` | μs | Std deviation of idle periods |
| `Idle Max` | μs | Maximum idle period duration |
| `Idle Min` | μs | Minimum idle period duration |

### 5.9 Subflow Statistics

A "subflow" is a subsequence of packets within a flow. These features aggregate packet and byte counts at the subflow level.

| Column | Description |
|--------|-------------|
| `Subflow Fwd Pkts` | Average forward packets per subflow |
| `Subflow Fwd Byts` | Average forward bytes per subflow |
| `Subflow Bwd Pkts` | Average backward packets per subflow |
| `Subflow Bwd Byts` | Average backward bytes per subflow |

### 5.10 Derived Ratios and Aggregates

| Column | Description | Formula (if applicable) |
|--------|-------------|------------------------|
| `Pkt Size Avg` | Average packet size overall | `Total Bytes / Total Packets` |
| `Fwd Act Data Pkts` | Forward packets with payload (not pure ACK) | Count of fwd packets with data |
| `Fwd PSH Flags` | Forward PSH flag count | Same as PSH Flag Cnt |
| `Fwd URG Flags` | Forward URG flag count | Same as URG Flag Cnt |

---

## 6. Example: One Flow from Start to Finish

Let's trace a real web request to see how the 84 features are computed.

### Raw Packets

```
Time     Src → Dst           Size  Flags   Notes
──────────────────────────────────────────────────────
0.0000   A → B [SYN]          66   SYN     TCP handshake start
0.0012   B → A [SYN, ACK]     66   SYN,ACK TCP handshake response
0.0025   A → B [ACK]          54   ACK     TCP handshake complete
0.0030   A → B [PSH, ACK]    220   PSH,ACK HTTP GET /
0.0100   B → A [PSH, ACK]    300   PSH,ACK HTTP 200 (headers)
0.0105   B → A [ACK]        1460   ACK     HTTP body (part 1)
0.0110   B → A [ACK]        1460   ACK     HTTP body (part 2)
0.0115   B → A [ACK]        1460   ACK     HTTP body (part 3)
0.0120   B → A [ACK]         984   ACK     HTTP body (part 4)
0.0125   A → B [ACK]          54   ACK     ACK for data
0.1000   A → B [FIN, ACK]     54   FIN,ACK Close connection
0.1005   B → A [FIN, ACK]     54   FIN,ACK Close confirmed
0.1010   A → B [ACK]          54   ACK     Last ACK
```

### 5-Tuple

| Field | Value |
|-------|-------|
| Src IP | A |
| Dst IP | B |
| Src Port | 54321 |
| Dst Port | 80 |
| Protocol | 6 (TCP) |

### Computed Features (Selected)

**Duration**: `0.1010 - 0.0000 = 101,000 μs`

**Forward packets**: 6 (SYN, ACK, PSH GET, ACK, FIN, ACK)
**Backward packets**: 7 (SYN-ACK, PSH 200, ACK×4, FIN-ACK)

**Forward bytes**: 66 + 54 + 220 + 54 + 54 + 54 = **502 bytes**
**Backward bytes**: 66 + 300 + 1460 + 1460 + 1460 + 984 + 54 = **5,784 bytes**

**Forward packet lengths**: [66, 54, 220, 54, 54, 54]
- `Fwd Pkt Len Max`: 220
- `Fwd Pkt Len Min`: 54
- `Fwd Pkt Len Mean`: 83.67

**Backward packet lengths**: [66, 300, 1460, 1460, 1460, 984, 54]
- `Bwd Pkt Len Max`: 1460
- `Bwd Pkt Len Min`: 54
- `Bwd Pkt Len Mean`: 826.29

**Forward IATs**: [0.0012, 0.0013, 0.0005, 0.0070, 0.0875, 0.0005]
- `Fwd IAT Mean`: 16,333 μs (heavily influenced by the big 87.5ms gap)
- `Fwd IAT Std`: 32,380 μs

**TCP Flags**:
- `SYN Flag Cnt`: 1 (first SYN)
- `ACK Flag Cnt`: 12 (all but the first SYN)
- `FIN Flag Cnt`: 2 (one each direction)
- `PSH Flag Cnt`: 3 (GET request, 200 response, 200 body)
- `RST Flag Cnt`: 0

**Ratios**:
- `Down/Up Ratio`: 7/6 ≈ 1.17
- `Fwd Pkts/s`: 6 / 0.101 = 59.4
- `Bwd Pkts/s`: 7 / 0.101 = 69.3

### What This Tells Us (Intuition)

```
- Short duration (101ms)          → Not a long download
- Bwd bytes >> Fwd bytes          → Response larger than request (normal web)
- SYN=1, FIN=2                    → Normal TCP handshake + teardown
- Fwd Pkt Len Mean=83, Bwd=826    → Small requests, large responses (HTTP)
- Down/Up Ratio ≈ 1.17            → More data back than sent (normal)
```

Everything about this flow says **normal benign web traffic**.

---

## 7. How This Connects to Our Dataset

The CIC-UNSW-NB15 dataset is a CSV where **each row is one flow** with features exactly as described above.

**From the source** (UNB website):

> "We used CICFlowMeter to extract the new set of features from the provided captured network traffic data by the UNSW-NB15."

So the original raw `.pcap` files (~100 GB) were processed through the exact pipeline described in Section 3, producing a CSV with ~3.5 million flows (later sampled to 448k for the 80:20 ratio).

### What's in Data.csv vs CICFlowMeter_out.csv

| File | Contents | Use |
|------|----------|-----|
| `CICFlowMeter_out.csv` | ALL extracted flows (~3.5M) + full 5-tuple | Original full data |
| `Data.csv` | 80:20 sampled subset (~448k) with **no IPs** | Our training data |
| `Label.csv` | Numerical labels for each row in Data.csv | Our labels |

### Feature Columns Present in Our Data

A sample row from Data.csv:

```
Flow Duration, Tot Fwd Pkts, Tot Bwd Pkts, TotLen Fwd Pkts, TotLen Bwd Pkts,
Fwd Pkt Len Max, Fwd Pkt Len Min, Fwd Pkt Len Mean, Fwd Pkt Len Std,
Bwd Pkt Len Max, Bwd Pkt Len Min, Bwd Pkt Len Mean, Bwd Pkt Len Std,
Fwd IAT Max, Fwd IAT Min, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Tot,
Bwd IAT Max, Bwd IAT Min, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Tot,
Flow IAT Max, Flow IAT Min, Flow IAT Mean, Flow IAT Std,
Active Mean, Active Std, Active Max, Active Min,
Idle Mean, Idle Std, Idle Max, Idle Min,
Fwd Pkts/s, Bwd Pkts/s, Down/Up Ratio,
Fwd PSH Flags, Fwd URG Flags,
FIN Flag Cnt, SYN Flag Cnt, RST Flag Cnt, PSH Flag Cnt,
ACK Flag Cnt, URG Flag Cnt, CWE Flag Count, ECE Flag Cnt,
Init Fwd Win Byts, Init Bwd Win Byts,
Fwd Seg Size Avg, Fwd Seg Size Min, Bwd Seg Size Avg,
Subflow Fwd Pkts, Subflow Fwd Byts, Subflow Bwd Pkts, Subflow Bwd Byts,
Pkt Size Avg, Pkt Len Mean, Pkt Len Std, Pkt Len Var,
Fwd Act Data Pkts,
Dst Port, Protocol,
Label
```

**Note:** IP addresses, source port, and Flow ID are excluded from Data.csv. They are available in `CICFlowMeter_out.csv`.

### Why Exclude IPs from Training?

1. **Generalization**: If the model learns "IP 192.168.1.5 = attacker," it won't work on new networks.
2. **Privacy**: Public datasets should not expose real IPs from the lab.
3. **Overfitting**: IPs are identifiers, not generalizable features.

However, `Dst Port` and `Protocol` are kept because:
- `Dst Port` tells us the **service** (80=HTTP, 22=SSH, 443=HTTPS...). Attack patterns differ by service.
- `Protocol` tells us TCP vs UDP vs ICMP. Different protocols carry different attacks.

---

## 8. How This Connects to Real-Time Deployment

When deploying our trained model on live traffic:

```
Live Network Interface
        │
        ▼
tcpdump -i eth0 -s 0 -w capture.pcap  (capture 60 seconds of traffic)
        │
        ▼
CICFlowMeter (Python library: cicflowmeter)
  - Reads pcap
  - Groups packets by 5-tuple
  - Computes all 84 features per flow
        │
        ▼
One or more flow records (each = 84 features)
        │
        ▼
Preprocessing (same as training: scale, encode)
        │
        ▼
Trained Model → Attack classification per flow
```

**Critical point**: The CICFlowMeter library during deployment MUST compute the same 84 features in the same way as the training data was computed. This is why we use the standard CICFlowMeter tool for both — to ensure compatibility.

### What You See vs What the Model Sees

| You see (human-readable) | Model sees (numbers) |
|-------------------------|---------------------|
| "A web request to google.com" | `{Duration: 120000, Fwd Pkts: 22, Bwd Pkts: 19, ...}` |
| "Someone scanning ports" | `{Duration: 5000, Fwd Pkts: 3, Bwd Pkts: 0, SYN: 3, RST: 3, ...}` |
| "A DDoS attack" | `{Duration: 100, Fwd Pkts/s: 50000, Pkt Len Mean: 40, ...}` |

The model never sees URLs, payloads, or content. It makes decisions purely based on the **statistical behavior** of each flow — how many packets, how fast, what sizes, what TCP flags, etc. This is both a strength (privacy-preserving, fast) and a limitation (can't analyze payload content like SQL injection strings).

---

## Appendix: Quick Reference — Key Features for Attack Detection

| Attack Type | Key Distinguishing Features |
|-------------|---------------------------|
| **DoS / DDoS** | Very high `Fwd Pkts/s`, very low `Pkt Len Mean`, low `Bwd Pkts`, short `Flow Duration` |
| **Port Scan** | High `SYN Flag Cnt`, high `RST Flag Cnt`, very low `Tot Bwd Pkts`, short `Flow Duration` |
| **Brute Force (SSH)** | Multiple flows to port 22, similar packet sizes, regular IAT (automated) |
| **Data Exfiltration** | High `Fwd Pkt Len Mean`, `Down/Up Ratio` skewed forward, long `Flow Duration` |
| **SQL Injection** | Unusual packet length patterns on port 80/443, unusual flag combinations |
| **Worms** | Many similar flows to different IPs, same port, same packet size profile |
| **Reconnaissance** | Very short flows, minimal data exchange, many unique destinations |
| **Normal Traffic** | Balanced `Down/Up Ratio`, varied IAT (human-like), proper TCP handshake |

---

*This document is meant for educational purposes. It explains the bridge between raw network packets and the statistical features our machine learning model uses for intrusion detection.*
