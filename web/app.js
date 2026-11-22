/**
 * EduPulse AI — Interactive Web Application Logic
 * Clean ES6+ • Real-time Data Visualization • Socratic Coach Preview
 */

const API_BASE = window.location.origin;

// State management
const state = {
  activeTab: 'overview',
  overviewData: null,
  charts: {},
  studentsPage: 1,
  studentsLimit: 25,
  filters: {
    search: '',
    module: 'all',
    presentation: 'all',
    risk: 'all',
    sortBy: 'id_student',
    sortDir: 'asc'
  },
  simDebounceTimer: null,
  searchDebounceTimer: null
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initOverview();
  initStudentsCohort();
  initSimulator();
  initCutoffs();
  initFeatures();
  initTextbookAI();
  initModal();
});

/* ==========================================================================
   Navigation & Tabs
   ========================================================================== */
function initNavigation() {
  const tabs = document.querySelectorAll('.nav-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetTab = tab.getAttribute('data-tab');
      switchTab(targetTab);
    });
  });

  // Global search enter key
  const globalSearch = document.getElementById('global-student-search');
  if (globalSearch) {
    globalSearch.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const query = globalSearch.value.trim();
        if (query) {
          switchTab('students');
          const studentFilterInput = document.getElementById('filter-search-id');
          if (studentFilterInput) {
            studentFilterInput.value = query;
            state.filters.search = query;
            state.studentsPage = 1;
            loadStudents();
          }
        }
      }
    });
  }
}

function switchTab(tabId) {
  state.activeTab = tabId;

  // Update tabs active state
  document.querySelectorAll('.nav-tab').forEach(t => {
    t.classList.toggle('active', t.getAttribute('data-tab') === tabId);
  });

  // Update panels
  document.querySelectorAll('.tab-panel').forEach(p => {
    p.classList.toggle('active', p.id === `panel-${tabId}`);
  });

  // Re-trigger layout adjustments for charts if needed
  if (tabId === 'overview' && state.overviewData) {
    Object.values(state.charts).forEach(c => c && c.resize && c.resize());
  }
}

/* ==========================================================================
   Tab 1: Overview & Analytics
   ========================================================================== */
async function initOverview() {
  try {
    const res = await fetch(`${API_BASE}/api/overview`);
    if (!res.ok) throw new Error('Failed to load overview data');
    const data = await res.json();
    state.overviewData = data;

    renderOverviewKPIs(data);
    renderComparativeGrid(data.comparative);
    renderCharts(data);
  } catch (err) {
    console.error('Error initializing overview:', err);
  }
}

function renderOverviewKPIs(data) {
  const totalElem = document.getElementById('val-total-enrolments');
  const riskRateElem = document.getElementById('val-at-risk-rate');
  const riskCountElem = document.getElementById('val-at-risk-count');
  const onTrackRateElem = document.getElementById('val-on-track-rate');
  const onTrackCountElem = document.getElementById('val-on-track-count');

  if (totalElem) totalElem.textContent = data.total_enrolments.toLocaleString();
  if (riskRateElem) riskRateElem.textContent = `${data.at_risk_prevalence}%`;
  if (riskCountElem) {
    riskCountElem.textContent = `${data.at_risk_count.toLocaleString()} students (Fail: ${data.outcomes.Fail?.toLocaleString()} + Withdrawn: ${data.outcomes.Withdrawn?.toLocaleString()})`;
  }
  if (onTrackRateElem) {
    const onTrackRate = (100 - data.at_risk_prevalence).toFixed(1);
    onTrackRateElem.textContent = `${onTrackRate}%`;
  }
  if (onTrackCountElem) {
    onTrackCountElem.textContent = `${data.on_track_count.toLocaleString()} students (Pass: ${data.outcomes.Pass?.toLocaleString()} + Distinction: ${data.outcomes.Distinction?.toLocaleString()})`;
  }
}

function renderComparativeGrid(comp) {
  const container = document.getElementById('comparative-metrics-grid');
  if (!container || !comp) return;

  const metrics = [
    { label: 'Avg VLE Clicks', key: 'vle_clicks_total', unit: '' },
    { label: 'Avg Active Days', key: 'vle_active_days', unit: ' days' },
    { label: 'Avg Assessment Score', key: 'mean_score', unit: ' / 100' },
    { label: 'Submission Rate', key: 'submission_rate', unit: '%' },
    { label: 'Days Since Activity', key: 'days_since_activity', unit: ' days' },
    { label: 'Late Submission Rate', key: 'late_submission_rate', unit: '%' }
  ];

  container.innerHTML = metrics.map(m => {
    const val = comp[m.key] || { safe: 0, risk: 0 };
    return `
      <div class="comp-metric-box">
        <span class="comp-metric-label">${m.label}</span>
        <div class="comp-metric-bars">
          <div class="comp-bar-row">
            <span class="comp-bar-tag safe">Safe</span>
            <span class="comp-bar-val safe-color">${val.safe}${m.unit}</span>
          </div>
          <div class="comp-bar-row">
            <span class="comp-bar-tag risk">At-Risk</span>
            <span class="comp-bar-val risk-color">${val.risk}${m.unit}</span>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

function renderCharts(data) {
  // Chart 1: Outcomes Donut
  const ctxOutcomes = document.getElementById('chart-outcomes');
  if (ctxOutcomes) {
    const outcomes = data.outcomes || {};
    state.charts.outcomes = new Chart(ctxOutcomes, {
      type: 'doughnut',
      data: {
        labels: ['Pass', 'Fail', 'Withdrawn', 'Distinction'],
        datasets: [{
          data: [
            outcomes.Pass || 0,
            outcomes.Fail || 0,
            outcomes.Withdrawn || 0,
            outcomes.Distinction || 0
          ],
          backgroundColor: ['#10b981', '#f43f5e', '#f59e0b', '#06b6d4'],
          borderColor: '#0a0e17',
          borderWidth: 2,
          hoverOffset: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 12 } }
          }
        },
        cutout: '68%'
      }
    });
  }

  // Chart 2: Module Risk Rate
  const ctxModules = document.getElementById('chart-modules');
  if (ctxModules) {
    const mods = data.module_stats || [];
    state.charts.modules = new Chart(ctxModules, {
      type: 'bar',
      data: {
        labels: mods.map(m => `Mod ${m.module}`),
        datasets: [{
          label: 'Risk Rate (%)',
          data: mods.map(m => m.risk_rate),
          backgroundColor: mods.map(m => m.risk_rate > 50 ? '#f43f5e' : '#06b6d4'),
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            max: 70,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', callback: val => `${val}%` }
          },
          x: {
            grid: { display: false },
            ticks: { color: '#94a3b8' }
          }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  }

  // Chart 3: Education Level Risk
  const ctxEdu = document.getElementById('chart-education');
  if (ctxEdu) {
    const edu = data.demographics?.education || [];
    state.charts.education = new Chart(ctxEdu, {
      type: 'bar',
      data: {
        labels: edu.map(e => e.category),
        datasets: [{
          label: 'At-Risk Rate (%)',
          data: edu.map(e => e.risk_rate),
          backgroundColor: '#3b82f6',
          borderRadius: 6
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            beginAtZero: true,
            max: 70,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', callback: val => `${val}%` }
          },
          y: {
            grid: { display: false },
            ticks: { color: '#94a3b8' }
          }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  }

  // Chart 4: VLE Engagement comparison
  const ctxEng = document.getElementById('chart-engagement');
  if (ctxEng) {
    const comp = data.comparative || {};
    state.charts.engagement = new Chart(ctxEng, {
      type: 'bar',
      data: {
        labels: ['Total Clicks', 'Active Days × 20', 'Submission %'],
        datasets: [
          {
            label: 'Safe Cohort',
            data: [
              comp.vle_clicks_total?.safe || 0,
              (comp.vle_active_days?.safe || 0) * 20,
              comp.submission_rate?.safe || 0
            ],
            backgroundColor: '#10b981',
            borderRadius: 6
          },
          {
            label: 'At-Risk Cohort',
            data: [
              comp.vle_clicks_total?.risk || 0,
              (comp.vle_active_days?.risk || 0) * 20,
              comp.submission_rate?.risk || 0
            ],
            backgroundColor: '#f43f5e',
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8' }
          },
          x: {
            grid: { display: false },
            ticks: { color: '#94a3b8' }
          }
        },
        plugins: {
          legend: {
            position: 'top',
            labels: { color: '#94a3b8' }
          }
        }
      }
    });
  }
}

/* ==========================================================================
   Tab 2: Student Risk Cohort
   ========================================================================== */
function initStudentsCohort() {
  const searchInput = document.getElementById('filter-search-id');
  const moduleSelect = document.getElementById('filter-module');
  const presSelect = document.getElementById('filter-presentation');
  const riskSelect = document.getElementById('filter-risk');
  const sortSelect = document.getElementById('filter-sort');
  const applyBtn = document.getElementById('btn-apply-filters');
  const resetBtn = document.getElementById('btn-reset-filters');
  const prevBtn = document.getElementById('btn-prev-page');
  const nextBtn = document.getElementById('btn-next-page');

  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      clearTimeout(state.searchDebounceTimer);
      state.searchDebounceTimer = setTimeout(() => {
        state.filters.search = e.target.value.trim();
        state.studentsPage = 1;
        loadStudents();
      }, 350);
    });
  }

  if (applyBtn) {
    applyBtn.addEventListener('click', () => {
      state.filters.module = moduleSelect.value;
      state.filters.presentation = presSelect.value;
      state.filters.risk = riskSelect.value;
      state.filters.sortBy = sortSelect.value;
      state.studentsPage = 1;
      loadStudents();
    });
  }

  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      searchInput.value = '';
      moduleSelect.value = 'all';
      presSelect.value = 'all';
      riskSelect.value = 'all';
      sortSelect.value = 'id_student';
      state.filters = {
        search: '',
        module: 'all',
        presentation: 'all',
        risk: 'all',
        sortBy: 'id_student',
        sortDir: 'asc'
      };
      state.studentsPage = 1;
      loadStudents();
    });
  }

  if (prevBtn) {
    prevBtn.addEventListener('click', () => {
      if (state.studentsPage > 1) {
        state.studentsPage--;
        loadStudents();
      }
    });
  }

  if (nextBtn) {
    nextBtn.addEventListener('click', () => {
      state.studentsPage++;
      loadStudents();
    });
  }

  // Initial load
  loadStudents();
}

async function loadStudents() {
  const tbody = document.getElementById('students-table-body');
  const pageIndicator = document.getElementById('page-indicator');
  const paginationInfo = document.getElementById('pagination-info-text');
  const prevBtn = document.getElementById('btn-prev-page');
  const nextBtn = document.getElementById('btn-next-page');

  if (tbody) {
    tbody.innerHTML = `<tr><td colspan="11" class="loading-cell">Loading student cohort records...</td></tr>`;
  }

  try {
    const params = new URLSearchParams({
      page: state.studentsPage,
      limit: state.studentsLimit,
      module: state.filters.module,
      presentation: state.filters.presentation,
      risk: state.filters.risk,
      sort_by: state.filters.sortBy,
      sort_dir: state.filters.sortDir
    });

    if (state.filters.search) {
      params.append('search', state.filters.search);
    }

    const res = await fetch(`${API_BASE}/api/students?${params.toString()}`);
    if (!res.ok) throw new Error('Failed to fetch students');
    const data = await res.json();

    renderStudentsTable(data.students);

    // Update pagination
    if (pageIndicator) pageIndicator.textContent = `Page ${data.page} of ${data.total_pages}`;
    if (paginationInfo) {
      const start = (data.page - 1) * data.limit + 1;
      const end = Math.min(data.page * data.limit, data.total);
      paginationInfo.textContent = `Showing ${start.toLocaleString()}–${end.toLocaleString()} of ${data.total.toLocaleString()} students`;
    }

    if (prevBtn) prevBtn.disabled = (data.page <= 1);
    if (nextBtn) nextBtn.disabled = (data.page >= data.total_pages);
  } catch (err) {
    console.error('Error loading students:', err);
    if (tbody) {
      tbody.innerHTML = `<tr><td colspan="11" class="loading-cell" style="color: var(--accent-rose);">Error loading student data. Check backend connection.</td></tr>`;
    }
  }
}

function renderStudentsTable(students) {
  const tbody = document.getElementById('students-table-body');
  if (!tbody) return;

  if (!students || students.length === 0) {
    tbody.innerHTML = `<tr><td colspan="11" class="loading-cell">No matching student records found.</td></tr>`;
    return;
  }

  tbody.innerHTML = students.map(s => {
    const isRisk = s.target_at_risk === 1;
    const scoreStr = (s.mean_score !== null && s.mean_score !== undefined) ? `${s.mean_score} / 100` : '<span style="color: var(--text-muted);">None due</span>';
    const inactiveStr = (s.vle_days_since_last_activity !== null && s.vle_days_since_last_activity !== undefined) ? `${s.vle_days_since_last_activity}d ago` : 'Never';
    const riskPill = isRisk
      ? `<span class="risk-pill at-risk">⚠️ ${s.final_result}</span>`
      : `<span class="risk-pill safe">✅ ${s.final_result}</span>`;

    return `
      <tr>
        <td>
          <a class="student-id-link" data-id="${s.id_student}">#${s.id_student}</a>
        </td>
        <td><strong>${s.code_module}</strong></td>
        <td>${s.code_presentation}</td>
        <td>${s.gender} &bull; ${s.highest_education}</td>
        <td>${scoreStr}</td>
        <td><strong>${(s.vle_clicks_total || 0).toLocaleString()}</strong></td>
        <td>${s.vle_active_days || 0}d</td>
        <td>${inactiveStr}</td>
        <td>${s.n_late_submissions > 0 ? `<span style="color: var(--accent-amber); font-weight: 600;">${s.n_late_submissions} late</span>` : '0'}</td>
        <td>${riskPill}</td>
        <td>
          <button class="btn btn-outline btn-sm inspect-btn" data-id="${s.id_student}">
            Inspect &bull; Coach
          </button>
        </td>
      </tr>
    `;
  }).join('');

  // Attach click handlers to open drawer
  tbody.querySelectorAll('.student-id-link, .inspect-btn').forEach(el => {
    el.addEventListener('click', () => {
      const studentId = el.getAttribute('data-id');
      openStudentModal(studentId);
    });
  });
}

/* ==========================================================================
   Tab 3: What-If Risk Simulator
   ========================================================================== */
function initSimulator() {
  const inputs = [
    'sim-clicks',
    'sim-active-days',
    'sim-days-inactive',
    'sim-score',
    'sim-submission-rate',
    'sim-late-rate',
    'sim-prev-attempts'
  ];

  inputs.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', () => {
      updateSliderDisplay(id, el.value);
      debouncedRunSimulation();
    });
  });

  const copyBtn = document.getElementById('btn-copy-sim-prompt');
  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      const promptText = document.getElementById('sim-coach-prompt')?.textContent || '';
      navigator.clipboard.writeText(promptText).then(() => {
        copyBtn.textContent = '✅ Copied!';
        setTimeout(() => copyBtn.textContent = '📋 Copy Message', 2000);
      });
    });
  }

  // Initial simulation
  runSimulation();
}

function updateSliderDisplay(id, val) {
  const displayMap = {
    'sim-clicks': () => `${parseInt(val, 10).toLocaleString()}`,
    'sim-active-days': () => `${val} days`,
    'sim-days-inactive': () => `${val} days ago`,
    'sim-score': () => `${val} / 100`,
    'sim-submission-rate': () => `${Math.round(val * 100)}%`,
    'sim-late-rate': () => `${Math.round(val * 100)}%`,
    'sim-prev-attempts': () => `${val} attempts`
  };

  const target = document.getElementById(`val-${id}`);
  if (target && displayMap[id]) {
    target.textContent = displayMap[id]();
  }
}

function debouncedRunSimulation() {
  clearTimeout(state.simDebounceTimer);
  state.simDebounceTimer = setTimeout(runSimulation, 120);
}

async function runSimulation() {
  const clicks = parseFloat(document.getElementById('sim-clicks')?.value || 350);
  const activeDays = parseFloat(document.getElementById('sim-active-days')?.value || 12);
  const inactiveDays = parseFloat(document.getElementById('sim-days-inactive')?.value || 2);
  const score = parseFloat(document.getElementById('sim-score')?.value || 68);
  const subRate = parseFloat(document.getElementById('sim-submission-rate')?.value || 1.0);
  const lateRate = parseFloat(document.getElementById('sim-late-rate')?.value || 0.0);
  const prevAttempts = parseInt(document.getElementById('sim-prev-attempts')?.value || 0, 10);

  try {
    const res = await fetch(`${API_BASE}/api/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        vle_clicks_total: clicks,
        vle_active_days: activeDays,
        vle_days_since_last_activity: inactiveDays,
        mean_score: score,
        assessment_submission_rate: subRate,
        late_submission_rate: lateRate,
        num_of_prev_attempts: prevAttempts
      })
    });

    if (!res.ok) throw new Error('Simulation failed');
    const data = await res.json();
    renderSimulationResults(data, { clicks, activeDays, inactiveDays, score, subRate, lateRate, prevAttempts });
  } catch (err) {
    console.error('Simulation error:', err);
  }
}

function renderSimulationResults(data, inputs) {
  const percentEl = document.getElementById('sim-risk-percent');
  const badgeEl = document.getElementById('sim-tier-badge');
  const adviceEl = document.getElementById('sim-advice-content');
  const gaugeCircle = document.getElementById('sim-gauge-circle');
  const factorsList = document.getElementById('sim-factors-list');

  const coachArchetype = document.getElementById('sim-coach-archetype');
  const coachAction = document.getElementById('sim-coach-action');
  const coachPrompt = document.getElementById('sim-coach-prompt');

  const prob = data.risk_probability;
  if (percentEl) percentEl.textContent = `${prob}%`;

  // Color mapping
  let color = '#10b981';
  let badgeClass = 'badge success';
  if (prob >= 70) {
    color = '#f43f5e';
    badgeClass = 'badge danger';
  } else if (prob >= 45) {
    color = '#f59e0b';
    badgeClass = 'badge';
  } else if (prob >= 25) {
    color = '#38bdf8';
    badgeClass = 'badge highlight';
  }

  if (badgeEl) {
    badgeEl.textContent = data.risk_tier;
    badgeEl.className = badgeClass;
    badgeEl.style.borderColor = color;
    badgeEl.style.color = color;
  }

  if (adviceEl) adviceEl.textContent = data.advisory_recommendation;

  // Gauge animation
  if (gaugeCircle) {
    const deg = Math.round((prob / 100) * 360);
    gaugeCircle.style.background = `conic-gradient(${color} 0deg, ${color} ${deg}deg, rgba(255, 255, 255, 0.08) ${deg}deg 360deg)`;
  }

  // Key factors
  if (factorsList && data.key_factors) {
    factorsList.innerHTML = data.key_factors.map(f => `
      <div class="factor-pill">
        <span class="factor-label">${f.factor}</span>
        <div class="factor-val">${f.value}</div>
      </div>
    `).join('');
  }

  // Socratic Coach dynamic responses
  if (coachArchetype && coachAction && coachPrompt) {
    if (inputs.clicks < 100 && inputs.inactiveDays > 10) {
      coachArchetype.textContent = 'Disengaged Ghosting Hazard';
      coachAction.textContent = 'Proactive advisor check-in via SMS. Verify tech connectivity and semester readiness.';
      coachPrompt.textContent = `"Hello! We noticed you haven't logged in over the last week. The upcoming Module 1 discussion is open, and we'd love to help you get back on track. What is one topic we can review together?"`;
    } else if (inputs.subRate < 0.8 || inputs.score < 50) {
      coachArchetype.textContent = 'Academic Remediation Required';
      coachAction.textContent = 'Assign guided Socratic scaffolding module for Assessment 1 core competencies.';
      coachPrompt.textContent = `"You made a great initial attempt on the assignment! Looking closely at Question 2, you were very close on the theoretical definition. How would you apply that same rule to the case study in Section B?"`;
    } else if (inputs.lateRate > 0) {
      coachArchetype.textContent = 'Time Management & Pacing Strain';
      coachAction.textContent = 'Provide 7-day milestone chunking template for next course deadline.';
      coachPrompt.textContent = `"Deadlines are moving quickly! To keep the pressure manageable for next week's submission, what time slot works best for you to complete your preliminary draft?"`;
    } else {
      coachArchetype.textContent = 'On-Track High Achiever';
      coachAction.textContent = 'Offer enrichment activities and maintain positive reinforcement.';
      coachPrompt.textContent = `"Solid cadence and strong assessment marks! Have you explored the extension exercises on the module discussion forum? Your perspective would be very valuable there!"`;
    }
  }
}

/* ==========================================================================
   Tab 4: Cutoff Diagnostics
   ========================================================================== */
async function initCutoffs() {
  try {
    const res = await fetch(`${API_BASE}/api/cutoff-diagnostics`);
    if (!res.ok) throw new Error('Failed to load cutoffs');
    const data = await res.json();

    renderCutoffTable(data.table);
    renderCutoffCards(data.analysis?.comparisons || []);
  } catch (err) {
    console.error('Error initializing cutoffs:', err);
  }
}

function renderCutoffTable(rows) {
  const tbody = document.getElementById('cutoff-table-body');
  if (!tbody || !rows) return;

  tbody.innerHTML = rows.map(r => {
    const isSelected = r.cutoff_day === 28;
    const remainingDays = 268 - r.cutoff_day;
    const statusTag = isSelected
      ? `<span class="badge success">Selected Benchmark</span>`
      : `<span class="badge">Reference Window</span>`;

    return `
      <tr style="${isSelected ? 'background: rgba(6, 182, 212, 0.08); font-weight: 600;' : ''}">
        <td><strong>Day ${r.cutoff_day}</strong></td>
        <td>${r.cohort_after_exclusions?.toLocaleString()}</td>
        <td>${r.presentations_with_assessment} / ${r.presentations_total}</td>
        <td>${r.pct_enrolments_with_ge1_graded}%</td>
        <td>${r.mean_graded_assessments}</td>
        <td>${r.vle_clicks_in_window_pct}%</td>
        <td>~${remainingDays} days remaining</td>
        <td>${statusTag}</td>
      </tr>
    `;
  }).join('');
}

function renderCutoffCards(cards) {
  const container = document.getElementById('cutoff-cards-grid');
  if (!container) return;

  container.innerHTML = cards.map(c => {
    const isDay28 = c.day === 28;
    return `
      <div class="cutoff-card ${isDay28 ? 'active' : ''}">
        <div class="cutoff-day-number">Day ${c.day}</div>
        <span class="cutoff-status-tag" style="color: ${isDay28 ? 'var(--brand-cyan)' : 'var(--text-secondary)'};">
          ${c.verdict}
        </span>
        <div class="cutoff-comment">
          ${c.issue}
        </div>
      </div>
    `;
  }).join('');
}

/* ==========================================================================
   Tab 5: Feature Catalog
   ========================================================================== */
async function initFeatures() {
  try {
    const res = await fetch(`${API_BASE}/api/features`);
    if (!res.ok) throw new Error('Failed to load features');
    const data = await res.json();

    const featureCols = data.feature_columns || [];
    renderFeaturesList(featureCols);

    const searchInput = document.getElementById('search-features-input');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        const filtered = featureCols.filter(col => col.toLowerCase().includes(query));
        renderFeaturesList(filtered);
      });
    }
  } catch (err) {
    console.error('Error initializing features:', err);
  }
}

function renderFeaturesList(features) {
  const container = document.getElementById('features-list-container');
  const countBadge = document.getElementById('features-counter-badge');
  if (!container) return;

  if (countBadge) countBadge.textContent = `${features.length} Predictors`;

  if (features.length === 0) {
    container.innerHTML = `<div style="grid-column: 1 / -1; padding: 24px; text-align: center; color: var(--text-muted);">No matching features found.</div>`;
    return;
  }

  // Feature family classifier helper
  function getFeatureFamily(name) {
    if (name.startsWith('vle_') || name === 'has_vle_activity') return 'VLE Activity';
    if (name.startsWith('assessment_') || name.startsWith('n_assessments_') || name.includes('score') || name.includes('submission')) return 'Assessment';
    if (name.includes('registration') || name.includes('registered')) return 'Registration';
    return 'Demographic';
  }

  container.innerHTML = features.map(col => {
    const family = getFeatureFamily(col);
    return `
      <div class="feature-item-card">
        <div class="feature-item-header">
          <span class="feature-name">${col}</span>
          <span class="feature-badge">${family}</span>
        </div>
        <div class="feature-desc">
          Strictly windowed to &le; day 28. Leakage verified.
        </div>
      </div>
    `;
  }).join('');
}

/* ==========================================================================
   Student Detail Modal / Drawer
   ========================================================================== */
function initModal() {
  const backdrop = document.getElementById('student-modal-backdrop');
  const closeBtn = document.getElementById('btn-close-modal');

  if (closeBtn) {
    closeBtn.addEventListener('click', closeModal);
  }

  if (backdrop) {
    backdrop.addEventListener('click', (e) => {
      if (e.target === backdrop) closeModal();
    });
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
  });
}

function closeModal() {
  const backdrop = document.getElementById('student-modal-backdrop');
  if (backdrop) backdrop.classList.remove('open');
}

async function openStudentModal(studentId) {
  const backdrop = document.getElementById('student-modal-backdrop');
  const titleEl = document.getElementById('modal-student-name');
  const metaTagsEl = document.getElementById('modal-meta-tags');
  const bodyEl = document.getElementById('modal-drawer-body');

  if (backdrop) backdrop.classList.add('open');
  if (titleEl) titleEl.textContent = `Student #${studentId}`;
  if (metaTagsEl) metaTagsEl.innerHTML = '';
  if (bodyEl) bodyEl.innerHTML = `<div class="loading-state">Loading detailed records for student #${studentId}...</div>`;

 
  try {
    const res = await fetch(`${API_BASE}/api/student/${studentId}`);
    if (!res.ok) throw new Error('Student profile not found');

    const data = await res.json();
    renderStudentDetail(data);

    // Request the trained model's prediction separately.
    const presentation = data.student.code_presentation;
    const predictionUrl =
      `${API_BASE}/api/model-predict/${studentId}` +
      `?presentation=${encodeURIComponent(presentation)}`;

    try {
      const predictionRes = await fetch(predictionUrl);
      const predictionData = await predictionRes.json();

      if (!predictionRes.ok) {
        throw new Error(
          predictionData.detail || 'Model prediction unavailable'
        );
      }

function renderModelPrediction(data) {
  const bodyEl = document.getElementById('modal-drawer-body');
  if (!bodyEl) return;

  const section = document.createElement('section');
  section.id = 'model-prediction-section';
  section.style.cssText =
    'margin-top: 20px; padding: 18px; border: 1px solid var(--brand-cyan); border-radius: 12px;';

  section.innerHTML = `
    <h3 class="drawer-section-title">Trained XGBoost Model Prediction</h3>
    <p>
      <strong>Predicted risk:</strong>
      ${Number(data.risk_probability).toFixed(1)}%
    </p>
    <p>
      <strong>Risk tier:</strong>
      ${data.risk_tier}
    </p>
    <p style="font-size: 12px; color: var(--text-secondary);">
      ${data.prediction_source}
    </p>
    <p style="font-size: 12px; color: var(--text-secondary);">
      Limitation: ${data.limitation}
    </p>
  `;

  // Replace an older prediction if the same modal is reused.
  document.getElementById('model-prediction-section')?.remove();
  bodyEl.appendChild(section);
}

function renderModelPredictionError(message) {
  const bodyEl = document.getElementById('modal-drawer-body');
  if (!bodyEl) return;

  document.getElementById('model-prediction-section')?.remove();

  const section = document.createElement('section');
  section.id = 'model-prediction-section';
  section.style.cssText = 'margin-top: 20px; padding: 18px;';
  section.textContent = message;

  bodyEl.appendChild(section);
}

      renderModelPrediction(predictionData);
    } catch (predictionError) {
      console.error('Model prediction error:', predictionError);
      renderModelPredictionError(
        'The trained model prediction is currently unavailable.'
      );
    }
    // Request the trained LSTM prediction separately.
    try {
      const lstmUrl =
        `${API_BASE}/api/lstm-predict/${studentId}` +
        `?presentation=${encodeURIComponent(presentation)}`;

      const lstmRes = await fetch(lstmUrl);
      const lstmData = await lstmRes.json();

      if (!lstmRes.ok) {
        throw new Error(
          lstmData.detail || 'LSTM prediction unavailable'
        );
      }

      const bodyEl = document.getElementById('modal-drawer-body');

      if (bodyEl) {
        document.getElementById('lstm-prediction-section')?.remove();

        const section = document.createElement('section');
        section.id = 'lstm-prediction-section';
        section.style.cssText =
          'margin-top: 20px; padding: 18px; border: 1px solid var(--brand-purple, #a78bfa); border-radius: 12px;';

        const heading = document.createElement('h3');
        heading.className = 'drawer-section-title';
        heading.textContent = 'Trained Weekly LSTM Model Prediction';

        const probability = document.createElement('p');
        probability.textContent =
          `Predicted risk: ${Number(lstmData.risk_probability).toFixed(1)}%`;

        const tier = document.createElement('p');
        tier.textContent = `Risk classification: ${lstmData.risk_tier}`;

        const source = document.createElement('p');
        source.style.cssText =
          'font-size: 12px; color: var(--text-secondary);';
        source.textContent = lstmData.prediction_source;

        const limitation = document.createElement('p');
        limitation.style.cssText =
          'font-size: 12px; color: var(--text-secondary);';
        limitation.textContent = `Limitation: ${lstmData.limitation}`;

        section.append(
          heading,
          probability,
          tier,
          source,
          limitation
        );

        bodyEl.appendChild(section);
      }
    } catch (lstmError) {
      console.error('LSTM prediction error:', lstmError);
    }
  } catch (err) {
    console.error('Error fetching student detail:', err);

    if (bodyEl) {
      bodyEl.innerHTML = `
        <div style="color: var(--accent-rose); padding: 20px;">
          Could not load student profile. Verify student ID.
        </div>`;
    }
  }
}


function renderStudentDetail(data) {
  const s = data.student;
  const p = data.risk_profile;
  const metaTagsEl = document.getElementById('modal-meta-tags');
  const bodyEl = document.getElementById('modal-drawer-body');

  const isRisk = p.is_at_risk;
  const statusPill = isRisk
    ? `<span class="risk-pill at-risk">⚠️ At-Risk (${p.final_outcome})</span>`
    : `<span class="risk-pill safe">✅ On-Track (${p.final_outcome})</span>`;

  if (metaTagsEl) {
    metaTagsEl.innerHTML = `
      <span class="badge">Module ${s.code_module}</span>
      <span class="badge">${s.code_presentation}</span>
      ${statusPill}
    `;
  }

  if (!bodyEl) return;

  bodyEl.innerHTML = `
    <!-- Top Summary Grid -->
    <div>
      <h3 class="drawer-section-title">Academic & Demographic Vitals</h3>
      <div class="drawer-stat-grid">
        <div class="drawer-stat-item">
          <span class="drawer-stat-label">Gender & Age</span>
          <div class="drawer-stat-val">${s.gender === 'F' ? 'Female' : 'Male'} &bull; ${s.age_band}</div>
        </div>
        <div class="drawer-stat-item">
          <span class="drawer-stat-label">Highest Education</span>
          <div class="drawer-stat-val">${s.highest_education}</div>
        </div>
        <div class="drawer-stat-item">
          <span class="drawer-stat-label">IMD Deprivation</span>
          <div class="drawer-stat-val">${s.imd_band || 'Not Reported'}</div>
        </div>
        <div class="drawer-stat-item">
          <span class="drawer-stat-label">Studied Credits</span>
          <div class="drawer-stat-val">${s.studied_credits} credits</div>
        </div>
        <div class="drawer-stat-item">
          <span class="drawer-stat-label">Prior Course Attempts</span>
          <div class="drawer-stat-val">${s.num_of_prev_attempts} attempt(s)</div>
        </div>
        <div class="drawer-stat-item">
          <span class="drawer-stat-label">Registration Timing</span>
          <div class="drawer-stat-val">${s.registration_days_before_start}d before start</div>
        </div>
      </div>
    </div>

    <!-- Day 28 Assessment Snapshot -->
    <div>
      <h3 class="drawer-section-title">Assessment Performance (Day &le; 28)</h3>
      <div class="drawer-stat-grid">
        <div class="drawer-stat-item">
          <span class="drawer-stat-label">Assessments Due / Graded</span>
          <div class="drawer-stat-val">${s.n_assessments_due || 0} due &bull; ${s.n_assessments_graded || 0} graded</div>
        </div>
        <div class="drawer-stat-item">
          <span class="drawer-stat-label">Mean Coursework Score</span>
          <div class="drawer-stat-val" style="color: ${s.mean_score < 40 ? 'var(--accent-rose)' : 'var(--brand-cyan)'};">
            ${s.mean_score !== null ? `${s.mean_score} / 100` : 'None Graded'}
          </div>
        </div>
        <div class="drawer-stat-item">
          <span class="drawer-stat-label">Late Submissions</span>
          <div class="drawer-stat-val" style="color: ${s.n_late_submissions > 0 ? 'var(--accent-amber)' : 'inherit'};">
            ${s.n_late_submissions || 0} late
          </div>
        </div>
      </div>
    </div>

    <!-- VLE Learning Platform Activity -->
    <div>
      <h3 class="drawer-section-title">VLE Platform Activity & Velocity</h3>
      <div class="drawer-stat-grid">
        <div class="drawer-stat-item">
          <span class="drawer-stat-label">Total Clicks</span>
          <div class="drawer-stat-val">${(s.vle_clicks_total || 0).toLocaleString()} clicks</div>
        </div>
        <div class="drawer-stat-item">
          <span class="drawer-stat-label">Active Study Days</span>
          <div class="drawer-stat-val">${s.vle_active_days || 0} of 28 days</div>
        </div>
        <div class="drawer-stat-item">
          <span class="drawer-stat-label">Days Inactive</span>
          <div class="drawer-stat-val">${s.vle_days_since_last_activity !== null ? `${s.vle_days_since_last_activity} days ago` : 'Never'}</div>
        </div>
        <div class="drawer-stat-item">
          <span class="drawer-stat-label">Pre-Course Clicks</span>
          <div class="drawer-stat-val">${s.vle_clicks_precourse || 0} clicks</div>
        </div>
        <div class="drawer-stat-item">
          <span class="drawer-stat-label">Forum Interactions</span>
          <div class="drawer-stat-val">${s.vle_clicks_forum || 0} clicks</div>
        </div>
        <div class="drawer-stat-item">
          <span class="drawer-stat-label">Content Resource Views</span>
          <div class="drawer-stat-val">${s.vle_clicks_content || 0} clicks</div>
        </div>
      </div>
    </div>

    <!-- Risk Drivers & Positive Signals -->
    <div>
      <h3 class="drawer-section-title">Empirical Risk Diagnostics</h3>
      ${p.risk_drivers.length > 0 ? `
        <div style="margin-bottom: 12px;">
          <span style="font-size: 11px; text-transform: uppercase; color: var(--accent-rose); font-weight: 700; letter-spacing: 0.5px;">Key Risk Drivers:</span>
          ${p.risk_drivers.map(d => `
            <div class="driver-item risk">
              <span>⚠️</span>
              <div>
                <strong>${d.factor}:</strong> ${d.detail}
              </div>
            </div>
          `).join('')}
        </div>
      ` : '<p style="font-size: 13px; color: var(--text-secondary); margin-bottom: 8px;">No major negative risk anomalies detected at cutoff day 28.</p>'}

      ${p.protective_factors.length > 0 ? `
        <div>
          <span style="font-size: 11px; text-transform: uppercase; color: var(--accent-emerald); font-weight: 700; letter-spacing: 0.5px;">Protective Signals:</span>
          ${p.protective_factors.map(pf => `
            <div class="driver-item safe">
              <span>✅</span>
              <div>
                <strong>${pf.factor}:</strong> ${pf.detail}
              </div>
            </div>
          `).join('')}
        </div>
      ` : ''}
    </div>

    <!-- Socratic Coach Blueprint -->
    <div class="glass-card" style="background: rgba(6, 182, 212, 0.05); border-color: rgba(6, 182, 212, 0.25);">
      <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">
        <span style="font-size: 18px;">✨</span>
        <h3 class="card-title" style="font-size: 16px;">AI Socratic Coaching Blueprint</h3>
      </div>

      <div style="display: flex; flex-direction: column; gap: 12px;">
        <div>
          <span class="coach-field-label">Student Persona Archetype</span>
          <div class="coach-field-value">${p.archetype}</div>
        </div>

        <div>
          <span class="coach-field-label">Strategic Objective</span>
          <div style="font-size: 13px; color: var(--text-primary);">${p.coaching_strategy}</div>
        </div>

        <div>
          <span class="coach-field-label">Recommended Tutor Action</span>
          <p style="font-size: 13px; color: var(--text-secondary);">${p.recommended_action}</p>
        </div>

        <div class="coach-prompt-box">
          <span class="prompt-label">Suggested Socratic Outreach Message:</span>
          <blockquote class="socratic-prompt" id="modal-socratic-msg">${p.socratic_coaching_prompt}</blockquote>
          <button class="btn btn-outline btn-sm copy-btn" id="btn-copy-modal-prompt">📋 Copy Message</button>
        </div>
      </div>
    </div>
  `;

  // Attach copy handler inside modal
  const copyModalBtn = document.getElementById('btn-copy-modal-prompt');
  if (copyModalBtn) {
    copyModalBtn.addEventListener('click', () => {
      const msg = document.getElementById('modal-socratic-msg')?.textContent || '';
      navigator.clipboard.writeText(msg).then(() => {
        copyModalBtn.textContent = '✅ Copied!';
        setTimeout(() => copyModalBtn.textContent = '📋 Copy Message', 2000);
      });
    });
  }
}



/* ==========================================================================
   Tab 7: Textbook AI Assistant
   ========================================================================== */
function initTextbookAI() {
  const askButton = document.getElementById('rag-ask-btn');
  const questionInput = document.getElementById('rag-question');
  const subjectSelect = document.getElementById('rag-subject');
  const statusElement = document.getElementById('rag-status');
  const answerContainer = document.getElementById('rag-answer-container');
  const answerElement = document.getElementById('rag-answer');
  const sourcesElement = document.getElementById('rag-sources');

  if (!askButton || !questionInput || !subjectSelect) {
    console.error('Textbook AI elements not found.');
    return;
  }

  askButton.addEventListener('click', async () => {
    const question = questionInput.value.trim();
    const subject = subjectSelect.value;

    if (!question) {
      statusElement.textContent = 'Please enter a question.';
      answerContainer.hidden = true;
      return;
    }

    askButton.disabled = true;
    askButton.textContent = 'Generating Answer...';
    statusElement.textContent = 'Searching your textbook and generating an answer...';
    answerContainer.hidden = true;

    try {
      const response = await fetch(`${API_BASE}/api/rag/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          question: question,
          subject: subject,
          top_k: 5
        })
      });

const responseText = await response.text();
let data = {};

try {
  data = responseText ? JSON.parse(responseText) : {};
} catch {
  throw new Error(
    `Server returned HTTP ${response.status}. ${responseText.slice(0, 200)}`
  );
}

if (!response.ok) {
  throw new Error(data.detail || 'Failed to generate an answer.');
}

      answerElement.textContent = data.answer || 'No answer received.';
      answerElement.style.whiteSpace = 'pre-wrap';

      answerContainer.hidden = false;

      sourcesElement.replaceChildren();

      (data.sources || []).forEach(source => {
        const sourceItem = document.createElement('div');
        sourceItem.textContent =
          `${source.source} — PDF page ${source.page}`;
        sourcesElement.appendChild(sourceItem);
      });

      statusElement.textContent = data.fallback
      ? 'AI providers unavailable. Showing retrieved textbook passages instead.'
      : `Answer generated successfully using ${data.provider || 'AI'}.`;

    } catch (error) {
      console.error('Textbook AI error:', error);
      statusElement.textContent = `Error: ${error.message}`;
    } finally {
      askButton.disabled = false;
      askButton.textContent = 'Ask Textbook AI';
    }
  });
}
