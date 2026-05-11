// k6 Load Testing Script for CARQ
// Run with: k6 run tests/load/load_test_k6.js

import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate, Trend, Counter, Gauge } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const responseTime = new Trend('response_time');
const ingestThroughput = new Counter('ingest_requests');
const concurrentUsers = new Gauge('concurrent_users');

// Test configuration
export const options = {
  stages: [
    // Ramp up to 100 users over 1 minute
    { duration: '1m', target: 100 },
    // Stay at 100 users for 5 minutes
    { duration: '5m', target: 100 },
    // Ramp down to 0 users over 1 minute
    { duration: '1m', target: 0 },
  ],
  thresholds: {
    // 95% of requests must complete below 5000ms
    'response_time': ['p(95)<5000'],
    // Error rate must stay below 5%
    'errors': ['rate<0.05'],
  },
  // Set user agent
  ext: {
    loadimpact: {
      projectID: 3356643,
      name: 'CARQ Load Test'
    }
  }
};

const BASE_URL = 'http://localhost:8000';

export default function () {
  // Set concurrent users gauge
  concurrentUsers.add(__VU);

  // Test 1: Health Check Endpoint
  group('Health Check', function () {
    const res = http.get(`${BASE_URL}/health`);
    
    const success = check(res, {
      'status is 200': (r) => r.status === 200,
      'response time < 1000ms': (r) => r.timings.duration < 1000,
    });
    
    errorRate.add(!success);
    responseTime.add(res.timings.duration);
  });

  // Test 2: Document Ingestion
  group('Document Ingestion', function () {
    const payload = JSON.stringify({
      file_path: `/tmp/test-${__VU}-${Date.now()}.pdf`,
      metadata: {
        source: 'k6_load_test',
        category: 'technical'
      },
      priority: 5
    });

    const params = {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer test-token'
      }
    };

    const res = http.post(`${BASE_URL}/api/v1/ingest`, payload, params);
    
    const success = check(res, {
      'status is 200 or 201': (r) => r.status === 200 || r.status === 201,
      'response time < 2000ms': (r) => r.timings.duration < 2000,
      'response has task_id': (r) => r.json('task_id') !== null
    });
    
    errorRate.add(!success);
    responseTime.add(res.timings.duration);
    ingestThroughput.add(1);
  });

  // Test 3: Search Similar Chunks
  group('Search Similar Chunks', function () {
    const payload = JSON.stringify({
      query: 'machine learning algorithms',
      limit: 10,
      similarity_threshold: 0.7,
      model: 'text-embedding-3-small'
    });

    const params = {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer test-token'
      }
    };

    const res = http.post(`${BASE_URL}/api/v1/search`, payload, params);
    
    const success = check(res, {
      'status is 200': (r) => r.status === 200,
      'response time < 3000ms': (r) => r.timings.duration < 3000,
      'response has results': (r) => r.json('results') !== null
    });
    
    errorRate.add(!success);
    responseTime.add(res.timings.duration);
  });

  // Test 4: Get Task Status
  group('Get Task Status', function () {
    const taskId = `task-${Math.floor(Math.random() * 1000)}`;
    
    const params = {
      headers: {
        'Authorization': 'Bearer test-token'
      }
    };

    const res = http.get(`${BASE_URL}/api/v1/tasks/${taskId}/status`, params);
    
    const success = check(res, {
      'status is 200': (r) => r.status === 200,
      'response time < 1000ms': (r) => r.timings.duration < 1000,
      'response has status field': (r) => r.json('status') !== null
    });
    
    errorRate.add(!success);
    responseTime.add(res.timings.duration);
  });

  // Test 5: Batch Embedding
  group('Batch Embedding', function () {
    const texts = [];
    for (let i = 0; i < 20; i++) {
      texts.push(`Document chunk number ${i} with various content`);
    }

    const payload = JSON.stringify({
      texts: texts,
      model: 'text-embedding-3-small'
    });

    const params = {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer test-token'
      }
    };

    const res = http.post(`${BASE_URL}/api/v1/embed/batch`, payload, params);
    
    const success = check(res, {
      'status is 200': (r) => r.status === 200,
      'response time < 4000ms': (r) => r.timings.duration < 4000,
      'response has embeddings': (r) => r.json('embeddings') !== null && r.json('embeddings').length > 0
    });
    
    errorRate.add(!success);
    responseTime.add(res.timings.duration);
  });

  // Think time between requests
  sleep(1);
}

// Function to simulate spike testing
export function spikeTest() {
  const res = http.get(`${BASE_URL}/health`);
  
  check(res, {
    'spike test - status is 200': (r) => r.status === 200,
  });
}

// Function to simulate soak testing
export function soakTest() {
  const res = http.post(
    `${BASE_URL}/api/v1/ingest`,
    JSON.stringify({
      file_path: `/tmp/soak-test-${Date.now()}.pdf`,
      metadata: { source: 'soak_test' },
      priority: 1
    }),
    {
      headers: { 'Content-Type': 'application/json' }
    }
  );
  
  check(res, {
    'soak test - status is ok': (r) => r.status === 200 || r.status === 201,
  });
  
  sleep(2);
}

// Performance threshold checks
export function handleSummary(data) {
  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    './tests/load/results.json': JSON.stringify(data),
  };
}

function textSummary(data, options) {
  let summary = '\n';
  summary += '═'.repeat(70) + '\n';
  summary += 'LOAD TEST SUMMARY\n';
  summary += '═'.repeat(70) + '\n';
  
  const metrics = data.metrics;
  
  if (metrics.response_time) {
    const rt = metrics.response_time.values;
    summary += `Response Time:\n`;
    summary += `  Min: ${rt.min.toFixed(2)}ms\n`;
    summary += `  Max: ${rt.max.toFixed(2)}ms\n`;
    summary += `  Avg: ${rt.avg.toFixed(2)}ms\n`;
    summary += `  P95: ${rt['p(95)'].toFixed(2)}ms\n`;
    summary += `  P99: ${rt['p(99)'].toFixed(2)}ms\n`;
  }
  
  if (metrics.errors) {
    summary += `\nError Rate: ${(metrics.errors.values.rate * 100).toFixed(2)}%\n`;
  }
  
  if (metrics.ingest_requests) {
    summary += `Ingest Throughput: ${metrics.ingest_requests.values.count} requests\n`;
  }
  
  summary += '═'.repeat(70) + '\n';
  
  return summary;
}
