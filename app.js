const API = 'http://localhost:8000';
const HORMONE_COLORS = {
  Cortisol:'#ef4444',Epinephrine:'#f97316',Dopamine:'#6366f1',
  Serotonin:'#06b6d4',Norepinephrine:'#eab308',Oxytocin:'#ec4899',
  EndogenousOpioids:'#10b981',GABA:'#8b5cf6',Glutamate:'#f43f5e',
  Endocannabinoids:'#14b8a6'
};
const EMOTIONS=[
  {name:'Joy',icon:'JOY',color:'#f59e0b',cat:'positive'},{name:'Fear',icon:'FEAR',color:'#ef4444',cat:'negative'},{name:'Anger',icon:'ANGER',color:'#dc2626',cat:'negative'},{name:'Sadness',icon:'SAD',color:'#3b82f6',cat:'negative'},
  {name:'Surprise',icon:'SURP',color:'#f97316',cat:'basic'},{name:'Disgust',icon:'DISG',color:'#65a30d',cat:'basic'},{name:'Love',icon:'LOVE',color:'#ec4899',cat:'social'},{name:'Anxiety',icon:'ANX',color:'#a855f7',cat:'negative'},
  {name:'Excitement',icon:'EXC',color:'#f59e0b',cat:'positive'},{name:'Calm',icon:'CALM',color:'#06b6d4',cat:'positive'},{name:'Guilt',icon:'GUILT',color:'#6366f1',cat:'social'},{name:'Jealousy',icon:'JEAL',color:'#e11d48',cat:'social'},
  {name:'Pride',icon:'PRIDE',color:'#eab308',cat:'positive'},{name:'Shame',icon:'SHAME',color:'#7c3aed',cat:'social'},{name:'Gratitude',icon:'GRAT',color:'#10b981',cat:'social'},{name:'Hope',icon:'HOPE',color:'#06b6d4',cat:'positive'},
  {name:'Loneliness',icon:'LONE',color:'#64748b',cat:'negative'},{name:'Frustration',icon:'FRUS',color:'#f97316',cat:'negative'},{name:'Relief',icon:'RELIEF',color:'#14b8a6',cat:'positive'},{name:'Nostalgia',icon:'NOST',color:'#8b5cf6',cat:'social'},
  {name:'Empathy',icon:'EMP',color:'#ec4899',cat:'social'},{name:'Boredom',icon:'BORE',color:'#94a3b8',cat:'negative'},{name:'Contentment',icon:'CONT',color:'#22c55e',cat:'positive'},{name:'Grief',icon:'GRIEF',color:'#1e3a5f',cat:'negative'}
];
const BRAIN_REGIONS=[
  {id:'prefrontal',name:'Prefrontal Cortex',x:155,y:105,w:90,h:60,color:'#6366f1',desc:'Decision making, planning, personality, working memory. Regulates impulse control and social behavior.',hormones:['Dopamine','Serotonin','GABA']},
  {id:'motor',name:'Motor Cortex',x:210,y:75,w:80,h:40,color:'#3b82f6',desc:'Voluntary movement control. Organizes and executes body movements via corticospinal tract.',hormones:['Glutamate','GABA','Dopamine']},
  {id:'somatosensory',name:'Somatosensory Cortex',x:260,y:85,w:80,h:40,color:'#0ea5e9',desc:'Processes touch, pressure, pain, and temperature from the body surface.',hormones:['Glutamate','GABA']},
  {id:'auditory',name:'Auditory Cortex',x:120,y:200,w:65,h:45,color:'#14b8a6',desc:'Processes auditory information. Sound perception, pitch, and rhythm analysis.',hormones:['Glutamate','GABA','Acetylcholine']},
  {id:'visual',name:'Visual Cortex',x:430,y:180,w:70,h:55,color:'#8b5cf6',desc:'Processes visual information. Object recognition, motion detection, and spatial awareness.',hormones:['Glutamate','GABA']},
  {id:'amygdala',name:'Amygdala',x:235,y:230,w:40,h:35,color:'#ef4444',desc:'Emotional processing center. Detects threats, processes fear, and attaches emotional significance to memories.',hormones:['Epinephrine','Norepinephrine','Cortisol','Glutamate']},
  {id:'hippocampus',name:'Hippocampus',x:195,y:270,w:55,h:35,color:'#06b6d4',desc:'Memory formation and spatial navigation. Converts short-term memories to long-term and contextualizes emotions.',hormones:['Glutamate','Cortisol','Serotonin']},
  {id:'hypothalamus',name:'Hypothalamus',x:270,y:260,w:45,h:35,color:'#10b981',desc:'Master regulator. Controls body temperature, hunger, thirst, sleep, and the entire endocrine system via the pituitary.',hormones:['Cortisol','Oxytocin','Epinephrine','Dopamine']},
  {id:'thalamus',name:'Thalamus',x:285,y:195,w:50,h:40,color:'#f59e0b',desc:'Sensory relay station. Routes all sensory information (except smell) to the appropriate cortical areas.',hormones:['Glutamate','GABA']},
  {id:'pituitary',name:'Pituitary Gland',x:305,y:305,w:35,h:28,color:'#a855f7',desc:'The master gland. Releases hormones that control the thyroid, adrenals, and reproductive organs.',hormones:['Cortisol','Oxytocin']},
  {id:'brainstem',name:'Brainstem',x:355,y:340,w:45,h:55,color:'#64748b',desc:'Controls breathing, heart rate, blood pressure. The bridge between brain and body for autonomic functions.',hormones:['Epinephrine','Norepinephrine','Serotonin']},
  {id:'cerebellum',name:'Cerebellum',x:420,y:310,w:75,h:65,color:'#14b8a6',desc:'Coordinates movement, balance, and fine motor control. Also involved in language and attention.',hormones:['GABA','Glutamate','Dopamine']},
  {id:'nacc',name:'Nucleus Accumbens',x:220,y:190,w:38,h:30,color:'#ec4899',desc:'The brain\'s reward center. Processes pleasure, motivation, and reinforcement learning.',hormones:['Dopamine','EndogenousOpioids','Endocannabinoids']},
  {id:'cingulate',name:'Cingulate Cortex',x:240,y:140,w:55,h:35,color:'#d946ef',desc:'Emotional regulation, pain processing, and error detection. Links emotions to behavior.',hormones:['Serotonin','Cortisol','EndogenousOpioids']}
];

let simulations=[];
let currentPage='dashboard';
let currentChartData=null;
let currentPhases=null;
let highlightedHormone=null;
let animFrame=null;
let animProgress=0;
let selectedEmotions=new Set();

// Navigation
document.querySelectorAll('.nav-btn').forEach(btn=>{
  btn.addEventListener('click',()=>{
    document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    const page=btn.dataset.page;
    document.querySelectorAll('.page').forEach(p=>p.classList.toggle('active',p.id===`page-${page}`));
    currentPage=page;
    if(page==='dashboard'){
      loadDashboard();
      // redraw chart after layout is visible (fixes hidden-canvas 0-width bug)
      if(currentChartData && currentPhases){
        requestAnimationFrame(()=>{requestAnimationFrame(()=>drawChartFrame(1));});
        if(animFrame) cancelAnimationFrame(animFrame);
        animProgress=1;
        setTimeout(()=>startIdleAnimation(),300);
      }
    }
    if(page==='brain')renderBrain();
    if(page==='eeg'){
      renderEEGRecent();
      if(!eegData&&simulations.length>0){
        const last=simulations[0];
        const type=last.type||'scenario';
        const title=last.title||'';
        triggerEEGFromSimulation(type==='scenario'?title:null,type==='emotion'?title:null,last.icon||'🧠');
      }
      if(eegLastProfile)renderEEGBandBars(eegLastProfile);
      eegStartDrawWhenReady();
    }
  });
});

async function api(path,opts={}){
  try{
    const r=await fetch(`${API}${path}`,{headers:{'Content-Type':'application/json'},...opts});
    if(!r.ok){
      let detail='Request failed';
      try{detail=(await r.json()).detail||detail;}catch(_){}
      throw new Error(detail);
    }
    return await r.json();
  }catch(e){
    console.warn('API:',e.message);
    return {error:true, message:e.message, valid:false};
  }
}

async function checkStatus(){
  const d=await api('/health');
  const el=document.getElementById('apiStatus');
  if(d && !d.error){el.classList.add('online');el.querySelector('span:last-child').textContent=d.hormone_model_ready?'ML Ready':'API Connected';document.getElementById('modelStatus').textContent=d.hormone_model_ready?'Trained':'Untrained';}
}

// Dashboard
async function loadDashboard(){
  const sims=await api('/simulations');
  if(sims && sims.error) return;
  simulations=sims?.simulations||[];
  document.getElementById('totalSims').textContent=simulations.length;
  renderRecent();
  if(eegLastProfile) renderDashboardBands(eegLastProfile);
}

function renderDashboardBands(profile){
  const container=document.getElementById('dashboardBandBars');
  if(!container||!profile)return;
  const bands=[
    {key:'delta',name:'Delta',range:'0.5-4 Hz',color:'#ef4444'},
    {key:'theta',name:'Theta',range:'4-8 Hz',color:'#f97316'},
    {key:'alpha',name:'Alpha',range:'8-13 Hz',color:'#10b981'},
    {key:'beta',name:'Beta',range:'13-30 Hz',color:'#3b82f6'},
    {key:'gamma',name:'Gamma',range:'30-45 Hz',color:'#8b5cf6'}
  ];
  const maxVal=Math.max(...bands.map(b=>profile[b.key]||0),0.01);
  container.innerHTML=bands.map(b=>{
    const val=profile[b.key]||0;
    const pct=(val/maxVal*100).toFixed(0);
    return`<div class="band-row"><span class="band-label" style="color:${b.color}">${b.name} <span style="color:var(--text-dim);font-weight:400;font-size:11px">${b.range}</span></span><div class="band-bar-wrap"><div class="band-bar" style="width:${pct}%;background:${b.color}"></div></div><span class="band-pct">${(val*100).toFixed(0)}%</span></div>`;
  }).join('');
}

function renderRecent(){
  const list=document.getElementById('recentList');
  if(!simulations.length){list.innerHTML='<p class="empty-state">No simulations yet. Try one from the Simulator tab!</p>';return;}
  list.innerHTML=simulations.slice(0,15).map(s=>`
    <div class="recent-item" data-id="${s.id}" data-profile='${JSON.stringify(s.profile||{}).replace(/'/g,"&apos;")}' data-type="${s.type}" data-title="${(s.title||'').replace(/"/g,'&quot;')}">
      <div class="recent-icon">${s.icon||'🧠'}</div>
      <div class="recent-info"><strong>${s.title}</strong><span>${s.hormone||''} · ${s.date||''}</span></div>
      <button class="recent-delete" title="Delete">&times;</button>
    </div>
  `).join('');
  list.querySelectorAll('.recent-item').forEach(item=>{
    item.addEventListener('click',e=>{
      if(e.target.closest('.recent-delete')){
        api(`/simulations/${item.dataset.id}`,{method:'DELETE'});
        simulations=simulations.filter(s=>s.id!=item.dataset.id);
        renderRecent();
        if(currentPage==='eeg')renderEEGRecent();
        document.getElementById('totalSims').textContent=simulations.length;
        return;
      }
      try{
        const profile=JSON.parse(item.dataset.profile);
        if(profile.hormones){
          showSavedSimulation(item.dataset.type,decodeURIComponent(item.dataset.title),profile);
        }
      }catch(ex){}
    });
  });
}

// This function DISPLAYS a saved simulation WITHOUT re-saving
function showSavedSimulation(type,title,profile){
  const container=type==='emotion'?document.getElementById('emotionResult'):document.getElementById('simResult');
  renderResult(container,title,profile);
  renderChart(profile.hormones,profile.phases);
  triggerEEGFromSimulation(type==='scenario'?title:null,type==='emotion'?title:null,'🧠');
  if(eegLastProfile) renderDashboardBands(eegLastProfile);
  if(type==='emotion'){
    document.querySelectorAll('.nav-btn').forEach(b=>b.classList.toggle('active',b.dataset.page==='emotions'));
    document.querySelectorAll('.page').forEach(p=>p.classList.toggle('active',p.id==='page-emotions'));
    currentPage='emotions';
  }else{
    document.querySelectorAll('.nav-btn').forEach(b=>b.classList.toggle('active',b.dataset.page==='simulator'));
    document.querySelectorAll('.page').forEach(p=>p.classList.toggle('active',p.id==='page-simulator'));
    currentPage='simulator';
    document.getElementById('scenarioInput').value=title;
  }
}

// ===================== ANIMATED CHART =====================
let chartHover=null;

function renderChart(hormones,phases,animate=true){
  currentChartData=hormones;currentPhases=phases;
  if(animate){animProgress=0;if(animFrame)cancelAnimationFrame(animFrame);animateChartIn();}
  else drawChartFrame(1);
}
function animateChartIn(){
  animProgress+=0.025;
  if(animProgress>=1){animProgress=1;drawChartFrame(1);startIdleAnimation();return;}
  drawChartFrame(easeOutCubic(animProgress));
  animFrame=requestAnimationFrame(animateChartIn);
}
function startIdleAnimation(){let t=0;function idle(){t+=0.015;drawChartFrame(1,t);animFrame=requestAnimationFrame(idle);}animFrame=requestAnimationFrame(idle);}
function easeOutCubic(x){return 1-Math.pow(1-x,3);}

function drawChartFrame(progress,idleTime=0){
  if(!currentChartData||!currentPhases)return;
  const canvas=document.getElementById('hormoneChart');
  if(!canvas) return;
  const parent=canvas.parentElement;
  if(!parent) return;
  const rect=parent.getBoundingClientRect();
  // if dashboard hidden, width is 0 - defer redraw
  if(rect.width < 50 || rect.height < 50){
    // keep data, will redraw when dashboard becomes visible
    return;
  }
  const ctx=canvas.getContext('2d');
  const dpr=window.devicePixelRatio||1;
  canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;
  ctx.scale(dpr,dpr);
  const W=rect.width,H=rect.height;
  const pad={top:24,right:24,bottom:44,left:52};
  const cW=W-pad.left-pad.right,cH=H-pad.top-pad.bottom;
  ctx.clearRect(0,0,W,H);

  ctx.strokeStyle='rgba(100,100,200,0.06)';ctx.lineWidth=1;
  for(let i=0;i<=10;i++){const y=pad.top+(cH/10)*i;ctx.beginPath();ctx.moveTo(pad.left,y);ctx.lineTo(W-pad.right,y);ctx.stroke();}
  ctx.fillStyle='#475569';ctx.font='10px JetBrains Mono,monospace';ctx.textAlign='right';
  for(let i=0;i<=10;i+=2)ctx.fillText(`${(1-i/10).toFixed(1)}`,pad.left-8,pad.top+(cH/10)*i+4);
  ctx.textAlign='center';ctx.font='10px Inter,sans-serif';
  const step=Math.max(1,Math.floor(currentPhases.length/8));
  currentPhases.forEach((p,i)=>{if(i%step===0||i===currentPhases.length-1)ctx.fillText(p,pad.left+(cW/(currentPhases.length-1))*i,H-pad.bottom+18);});

  const entries=Object.entries(currentChartData);
  entries.forEach(([name,values])=>{
    const color=HORMONE_COLORS[name]||'#64748b';
    const isHl=highlightedHormone===name;
    const isDim=highlightedHormone&&!isHl;
    const alpha=isDim?0.12:1;
    const lineW=isHl?3.5:2;
    ctx.globalAlpha=alpha;
    const points=values.map((v,i)=>{
      const x=pad.left+(cW/(values.length-1))*i;
      const drawCount=Math.floor(progress*values.length);
      if(i>drawCount)return null;
      const w=Math.sin(idleTime*2+i*0.5+entries.findIndex(e=>e[0]===name)*0.7)*0.008;
      return{x,y:pad.top+cH*(1-(v+w))};
    }).filter(Boolean);
    if(points.length<2)return;
    ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=isHl?10:5;ctx.globalAlpha=alpha*(isHl?0.35:0.1);ctx.lineJoin='round';ctx.lineCap='round';
    points.forEach((p,i)=>i===0?ctx.moveTo(p.x,p.y):ctx.lineTo(p.x,p.y));ctx.stroke();
    ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=lineW;ctx.globalAlpha=alpha;
    points.forEach((p,i)=>i===0?ctx.moveTo(p.x,p.y):ctx.lineTo(p.x,p.y));ctx.stroke();
    if(progress>=1&&points.length>0){const last=points[points.length-1];const pulse=1+Math.sin(idleTime*4)*0.3;ctx.beginPath();ctx.arc(last.x,last.y,4*pulse,0,Math.PI*2);ctx.fillStyle=color;ctx.globalAlpha=alpha*0.6;ctx.fill();ctx.beginPath();ctx.arc(last.x,last.y,2,0,Math.PI*2);ctx.fillStyle=color;ctx.globalAlpha=alpha;ctx.fill();}
    if(points.length>1){ctx.beginPath();ctx.moveTo(points[0].x,pad.top+cH);points.forEach(p=>ctx.lineTo(p.x,p.y));ctx.lineTo(points[points.length-1].x,pad.top+cH);ctx.closePath();const g=ctx.createLinearGradient(0,pad.top,0,pad.top+cH);g.addColorStop(0,color+'18');g.addColorStop(1,color+'02');ctx.fillStyle=g;ctx.globalAlpha=alpha*0.5;ctx.fill();}
    ctx.globalAlpha=1;
  });

  if(chartHover!==null){
    const hx=pad.left+(cW/(currentPhases.length-1))*chartHover;
    ctx.beginPath();ctx.strokeStyle='rgba(148,163,184,0.3)';ctx.lineWidth=1;ctx.setLineDash([4,4]);ctx.moveTo(hx,pad.top);ctx.lineTo(hx,pad.top+cH);ctx.stroke();ctx.setLineDash([]);
    entries.forEach(([name,values])=>{if(chartHover<values.length){const v=values[chartHover];const c=HORMONE_COLORS[name]||'#64748b';const y=pad.top+cH*(1-v);ctx.beginPath();ctx.arc(hx,y,5,0,Math.PI*2);ctx.fillStyle=c;ctx.globalAlpha=highlightedHormone===name?1:0.7;ctx.fill();ctx.beginPath();ctx.arc(hx,y,2.5,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();ctx.globalAlpha=1;}});
  }

  const legend=document.getElementById('chartLegend');
  legend.innerHTML=entries.map(([name])=>{const isH=highlightedHormone===name;const isD=highlightedHormone&&!isH;return`<div class="legend-item${isH?' active':''}" data-hormone="${name}" style="opacity:${isD?0.3:1}"><span class="legend-dot" style="background:${HORMONE_COLORS[name]};${isH?'box-shadow:0 0 8px '+HORMONE_COLORS[name]:''}"></span>${name}</div>`;}).join('');
  legend.querySelectorAll('.legend-item').forEach(item=>{item.addEventListener('mouseenter',()=>{highlightedHormone=item.dataset.hormone;drawChartFrame(1,animFrame?performance.now()/1000:0);});item.addEventListener('mouseleave',()=>{highlightedHormone=null;drawChartFrame(1,animFrame?performance.now()/1000:0);});});
}

const chartContainer=document.querySelector('.chart-container');
if(chartContainer){chartContainer.addEventListener('mousemove',e=>{if(!currentPhases||!currentChartData)return;const canvas=document.getElementById('hormoneChart');const rect=canvas.getBoundingClientRect();const pad={left:52,right:24};const cW=rect.width-pad.left-pad.right;const ratio=(e.clientX-rect.left-pad.left)/cW;chartHover=Math.max(0,Math.min(currentPhases.length-1,Math.round(ratio*(currentPhases.length-1))));drawChartFrame(1,animFrame?performance.now()/1000:0);});chartContainer.addEventListener('mouseleave',()=>{chartHover=null;drawChartFrame(1,animFrame?performance.now()/1000:0);});}

// ===================== SIMULATOR =====================
document.getElementById('simulateBtn').addEventListener('click',()=>runScenario(document.getElementById('scenarioInput').value));
document.querySelectorAll('.tag-btn').forEach(btn=>{btn.addEventListener('click',()=>{document.getElementById('scenarioInput').value=btn.dataset.scenario;runScenario(btn.dataset.scenario);});});

async function runScenario(text){
  if(!text.trim())return;
  const result=await api('/hormone/predict',{method:'POST',body:JSON.stringify({scenario:text})});
  if(!result)return;
  if(result.error){
    document.getElementById('simResult').innerHTML=`
      <div class="result-placeholder" style="border:1px solid rgba(239,68,68,0.3);background:rgba(239,68,68,0.05);border-radius:12px;padding:32px;">
        <div class="placeholder-icon" style="color:#ef4444;">
          <svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.5" width="48" height="48"><circle cx="32" cy="32" r="28"/><line x1="32" y1="20" x2="32" y2="36"/><circle cx="32" cy="44" r="2"/></svg>
        </div>
        <p style="color:#ef4444;margin-top:12px;font-weight:500;">${result.message}</p>
        <p style="color:var(--text-dim);margin-top:8px;font-size:13px;">Describe a scenario involving emotions or personal experiences. For example:</p>
        <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:12px;justify-content:center;">
          <button class="tag-btn" onclick="document.getElementById('scenarioInput').value='You narrowly avoid a car accident';runScenario('You narrowly avoid a car accident')">Near-miss accident</button>
          <button class="tag-btn" onclick="document.getElementById('scenarioInput').value='Your best friend gives you an unexpected gift';runScenario('Your best friend gives you an unexpected gift')">Unexpected gift</button>
          <button class="tag-btn" onclick="document.getElementById('scenarioInput').value='You feel anxious about an upcoming exam';runScenario('You feel anxious about an upcoming exam')">Anxiety</button>
          <button class="tag-btn" onclick="document.getElementById('scenarioInput').value='You hug your partner after a long time apart';runScenario('You hug your partner after a long time apart')">Reunion hug</button>
        </div>
      </div>`;
    return;
  }
  document.getElementById('scenarioInput').value=text;
  await saveSim({title:text.slice(0,50),description:text,icon:'🧠',hormone:result.primary_hormone,profile:result,type:'scenario'});
  renderResult(document.getElementById('simResult'),text,result);
  renderChart(result.hormones,result.phases);
  triggerEEGFromSimulation(text,null,'🧠');
  if(eegLastProfile) renderDashboardBands(eegLastProfile);
}

// ===================== EMOTIONS =====================
function initEmotions(){
  const grid=document.getElementById('emotionGrid');
  // direct grid of 24 emotions - no searching/filtering, just different emotions
  grid.innerHTML=EMOTIONS.map(e=>`<div class="emotion-card" data-emotion="${e.name}" style="border-top:3px solid ${e.color}" title="${e.cat}"><span class="emotion-icon">${e.icon}</span><span class="emotion-name">${e.name}</span></div>`).join('');
  const comboBar=document.createElement('div');comboBar.className='emotion-combo-bar';
  comboBar.innerHTML=`<span class="combo-count" id="emotionCount">0 selected</span><button class="btn-primary" id="simulateComboBtn" disabled><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5,3 19,12 5,21"/></svg>Simulate Combination</button><button class="btn-secondary" id="clearComboBtn">Clear</button>`;
  grid.parentElement.insertBefore(comboBar,grid.nextSibling);
  grid.querySelectorAll('.emotion-card').forEach(card=>{
    card.addEventListener('click',()=>{
      const emotion=card.dataset.emotion;
      if(selectedEmotions.has(emotion)){selectedEmotions.delete(emotion);card.classList.remove('selected');}
      else{selectedEmotions.add(emotion);card.classList.add('selected');}
      updateComboBar();
      // single-click immediate simulation
      runEmotion(emotion);
    });
  });
  document.getElementById('simulateComboBtn').addEventListener('click',()=>runComboEmotions([...selectedEmotions]));
  document.getElementById('clearComboBtn').addEventListener('click',()=>{selectedEmotions.clear();grid.querySelectorAll('.emotion-card').forEach(c=>c.classList.remove('selected'));updateComboBar();});
}
function updateComboBar(){document.getElementById('emotionCount').textContent=`${selectedEmotions.size} selected`;document.getElementById('simulateComboBtn').disabled=selectedEmotions.size===0;}

async function runEmotion(name){
  const result=await api('/hormone/predict',{method:'POST',body:JSON.stringify({emotion:name})});
  if(!result)return;
  if(result.error){
    document.getElementById('emotionResult').innerHTML=`
      <div class="result-placeholder" style="border:1px solid rgba(239,68,68,0.3);background:rgba(239,68,68,0.05);border-radius:12px;padding:32px;">
        <p style="color:#ef4444;font-weight:500;">${result.message}</p>
      </div>`;
    return;
  }
  const emotion=EMOTIONS.find(e=>e.name===name);
  await saveSim({title:name,description:name,icon:emotion?.icon||'🧠',hormone:result.primary_hormone,profile:result,type:'emotion'});
  renderResult(document.getElementById('emotionResult'),`${emotion?.icon||''} ${name}`,result);
  renderChart(result.hormones,result.phases);
  triggerEEGFromSimulation(null,name,emotion?.icon||'🧠');
  if(eegLastProfile) renderDashboardBands(eegLastProfile);
}

async function runEmotionScenario(text){
  if(!text.trim())return;
  const result=await api('/hormone/predict',{method:'POST',body:JSON.stringify({scenario:text})});
  if(!result)return;
  if(result.error){
    document.getElementById('emotionResult').innerHTML=`
      <div class="result-placeholder" style="border:1px solid rgba(239,68,68,0.3);background:rgba(239,68,68,0.05);border-radius:12px;padding:32px;">
        <div class="placeholder-icon" style="color:#ef4444;">
          <svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.5" width="48" height="48"><circle cx="32" cy="32" r="28"/><line x1="32" y1="20" x2="32" y2="36"/><circle cx="32" cy="44" r="2"/></svg>
        </div>
        <p style="color:#ef4444;margin-top:12px;font-weight:500;">${result.message}</p>
        <p style="color:var(--text-dim);margin-top:8px;font-size:13px;">Describe a scenario involving emotions or personal experiences.</p>
      </div>`;
    return;
  }
  await saveSim({title:text.slice(0,50),description:text,icon:'🧠',hormone:result.primary_hormone,profile:result,type:'emotion'});
  renderResult(document.getElementById('emotionResult'),text,result);
  renderChart(result.hormones,result.phases);
  triggerEEGFromSimulation(text,null,'🧠');
  if(eegLastProfile) renderDashboardBands(eegLastProfile);
}

async function runComboEmotions(emotions){
  if(!emotions.length)return;
  const results=await Promise.all(emotions.map(e=>api('/hormone/predict',{method:'POST',body:JSON.stringify({emotion:e})})));
  const valid=results.filter(r=>r && !r.error && r.hormones);if(!valid.length)return;
  const combined={};const hormones=Object.keys(valid[0].hormones);
  hormones.forEach(h=>{combined[h]=valid[0].hormones[h].map((_,i)=>valid.reduce((sum,r)=>sum+r.hormones[h][i],0)/valid.length);});
  const primary=valid.reduce((best,r)=>{const peakBest=Math.max(...combined[best]);const peakCur=Math.max(...combined[r.primary_hormone]);return peakCur>peakBest?r.primary_hormone:best;},valid[0].primary_hormone);
  const comboResult={hormones:combined,primary_hormone:primary,pattern:`Combined ${emotions.join(' + ')}: overlapping hormone responses`,confidence:valid.reduce((s,r)=>s+r.confidence,0)/valid.length,scenario_type:'combined',phases:valid[0].phases,features:{}};
  const icons=emotions.map(e=>EMOTIONS.find(em=>em.name===e)?.icon||'').join(' ');
  const title=emotions.join(' + ');
  await saveSim({title,description:title,icon:icons,hormone:primary,profile:comboResult,type:'emotion'});
  renderResult(document.getElementById('emotionResult'),`${icons} ${title}`,comboResult);
  renderChart(combined,valid[0].phases);
  triggerEEGFromSimulation(null,emotions[0],icons||'🧠');
  if(eegLastProfile) renderDashboardBands(eegLastProfile);
}

function renderResult(container,trigger,result){
  const sorted=Object.entries(result.hormones).sort((a,b)=>Math.max(...b[1])-Math.max(...a[1]));
  container.innerHTML=`
    <div class="result-header"><h2>${trigger}</h2><span class="result-badge primary">${result.primary_hormone}</span><span class="result-badge confidence">${(result.confidence*100).toFixed(1)}% confidence</span></div>
    <div class="result-chart-wrap" style="height:260px; margin:16px 0; background:rgba(0,0,0,0.2); border:1px solid var(--border); border-radius:12px; padding:12px; position:relative">
      <canvas class="resultChart" style="width:100%;height:100%;display:block"></canvas>
    </div>
    <div class="hormone-grid">${sorted.map(([name,values])=>{const peak=Math.max(...values);const peakIdx=values.indexOf(peak);const change=((values[7]-values[0])/values[0]*100).toFixed(0);const color=HORMONE_COLORS[name]||'#64748b';return`<div class="hormone-card" data-hormone="${name}" style="border-left:3px solid ${color}"><div class="hormone-name">${name}</div><div class="hormone-value" style="color:${color}">${(peak*100).toFixed(0)}%</div><div style="font-size:11px;color:var(--text-dim);margin-top:2px">Peak at phase ${peakIdx}</div><div style="font-size:11px;color:${parseInt(change)>0?'var(--green)':'var(--rose)'};margin-top:2px">${parseInt(change)>0?'↑':'↓'} ${Math.abs(change)}% from baseline</div><svg class="hormone-spark" viewBox="0 0 100 24" preserveAspectRatio="none"><polyline fill="none" stroke="${color}" stroke-width="1.5" stroke-linejoin="round" points="${values.map((v,i)=>`${(i/(values.length-1))*100},${24-v*24}`).join(' ')}"/></svg></div>`;}).join('')}</div>
    <div class="result-details">
      <div class="detail-card"><span>Pattern</span><b>${result.pattern}</b></div>
      <div class="detail-card"><span>Scenario Type</span><b>${(result.scenario_type||'unknown').replace(/_/g,' ')}</b></div>
      <div class="detail-card"><span>Primary Hormone Mechanism</span><p>${getMechanism(result.primary_hormone)}</p></div>
      <div class="detail-card"><span>Biological Notes</span><p>${getNotes(result.primary_hormone)}</p></div>
    </div>
    <div style="margin-top:16px;text-align:center"><button class="btn-secondary" onclick="document.querySelector('[data-page=dashboard]').click()">View Full Timeline Chart on Dashboard →</button></div>`;
  // draw inline chart for immediate feedback
  const inlineCanvas=container.querySelector('.resultChart');
  if(inlineCanvas) drawInlineChart(inlineCanvas, result.hormones, result.phases);
}

function drawInlineChart(canvas, hormones, phases){
  if(!canvas||!hormones||!phases) return;
  const ctx=canvas.getContext('2d');
  const dpr=window.devicePixelRatio||1;
  const wrap=canvas.parentElement;
  const rect=wrap.getBoundingClientRect();
  canvas.width=rect.width*dpr; canvas.height=rect.height*dpr;
  ctx.scale(dpr,dpr);
  const W=rect.width, H=rect.height;
  const pad={top:16,right:16,bottom:32,left:40};
  const cW=W-pad.left-pad.right, cH=H-pad.top-pad.bottom;
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle='rgba(100,100,200,0.08)'; ctx.lineWidth=1;
  for(let i=0;i<=4;i++){const y=pad.top+(cH/4)*i; ctx.beginPath(); ctx.moveTo(pad.left,y); ctx.lineTo(W-pad.right,y); ctx.stroke();}
  ctx.fillStyle='#475569'; ctx.font='9px JetBrains Mono,monospace'; ctx.textAlign='right';
  for(let i=0;i<=4;i++) ctx.fillText((1-i/4).toFixed(1), pad.left-8, pad.top+(cH/4)*i+3);
  ctx.textAlign='center'; ctx.font='9px Inter,sans-serif';
  const step=Math.max(1,Math.floor(phases.length/4));
  phases.forEach((p,i)=>{if(i%step===0||i===phases.length-1) ctx.fillText(p, pad.left+(cW/(phases.length-1))*i, H-pad.bottom+14);});
  Object.entries(hormones).forEach(([name,values])=>{
    const color=HORMONE_COLORS[name]||'#64748b';
    ctx.beginPath(); ctx.strokeStyle=color; ctx.lineWidth=1.8; ctx.globalAlpha=0.9; ctx.lineJoin='round';
    values.forEach((v,i)=>{
      const x=pad.left+(cW/(values.length-1))*i;
      const y=pad.top+cH*(1-v);
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    });
    ctx.stroke(); ctx.globalAlpha=1;
  });
}

function getMechanism(h){const m={Cortisol:'HPA axis: Hypothalamus→CRH→Pituitary→ACTH→Adrenal Cortex→Cortisol. Peak at 20-30min.',Epinephrine:'SAM system: Sympathetic nerves→Adrenal Medulla→Epinephrine. Effects in seconds.',Dopamine:'Mesolimbic pathway: VTA→Nucleus Accumbens. Reward prediction error signal.',Serotonin:'Raphe nuclei projections. Modulates mood, sleep, appetite.',Norepinephrine:'Locus coeruleus→widespread arousal. Attention and vigilance.',Oxytocin:'Hypothalamus→Posterior pituitary. Social bonding and trust.',EndogenousOpioids:'Endorphin release from pituitary and hypothalamus. Pain relief and pleasure.',GABA:'Interneurons release GABA. Primary inhibitory neurotransmitter.',Glutamate:'Pyramidal neurons release glutamate. Primary excitatory neurotransmitter.',Endocannabinoids:'Retrograde signaling from postsynaptic neurons. Modulates release.'};return m[h]||'Context-dependent.';}
function getNotes(h){const n={Cortisol:'Diurnal rhythm peaks AM. Chronic elevation suppresses immunity.',Epinephrine:'Adrenaline. ↑HR, dilates airways, redirects blood to muscles.',Dopamine:'Prediction error signal. Peaks on novel reward, habituates.',Serotonin:'90% produced in gut. Brain serotonin modulates mood indirectly.',Norepinephrine:'Co-released with epinephrine. Focus on attention/vigilance.',Oxytocin:'Context-dependent. In-group bonding, can increase out-group distrust.',EndogenousOpioids:'Runner\'s high. Interact with mu-opioid receptors.',GABA:'Target of benzodiazepines. Reduces neuronal excitability.',Glutamate:'Excess→excitotoxicity. Brain balances GABA/Glutamate ratio.',Endocannabinoids:'Anandamide, 2-AG. Retrograde modulation of neurotransmission.'};return n[h]||'Individual responses vary.';}

// ===================== 3D ROTATING BRAIN =====================
let brainRotX=-15,brainRotY=0,brainAutoRotate=true,brainDragging=false,brainLastX=0,brainLastY=0;

function renderBrain(){
  const wrap=document.querySelector('.brain-svg-wrap');
  wrap.innerHTML=`
    <div class="brain3d-scene" id="brain3d">
      <div class="brain3d-stage" id="brainStage"></div>
    </div>
    <div class="brain-rotation-hint">Drag to rotate · Click region for details</div>
  `;
  const stage=document.getElementById('brainStage');
  const regionHTML=BRAIN_REGIONS.map(r=>`
    <div class="brain3d-region" data-id="${r.id}" style="
      left:${r.x/600*100}%;top:${r.y/420*100}%;
      width:${r.w/600*100}%;height:${r.h/420*100}%;
      --rc:${r.color};
      background:radial-gradient(ellipse,${r.color}55 0%,${r.color}15 60%,transparent 100%);
      border:1.5px solid ${r.color}88;
      border-radius:50%;
      position:absolute;
      transform:translate(-50%,-50%);
      cursor:pointer;
      transition:all .3s ease;
      display:flex;align-items:center;justify-content:center;
    ">
      <span class="brain3d-label" style="
        position:absolute;bottom:-18px;left:50%;transform:translateX(-50%);
        font-size:9px;color:#cbd5e1;white-space:nowrap;
        text-shadow:0 1px 4px rgba(0,0,0,0.8);
        pointer-events:none;font-family:Inter,sans-serif;
      ">${r.name}</span>
    </div>
  `).join('');
  stage.innerHTML=regionHTML;

  // Brain outline (ellipse)
  const outline=document.createElement('div');
  outline.className='brain3d-outline';
  stage.prepend(outline);

  // Click handlers
  stage.querySelectorAll('.brain3d-region').forEach(el=>{
    el.addEventListener('click',()=>{
      stage.querySelectorAll('.brain3d-region').forEach(r=>r.classList.remove('active'));
      el.classList.add('active');
      const region=BRAIN_REGIONS.find(r=>r.id===el.dataset.id);
      if(!region)return;
      document.getElementById('brainInfo').innerHTML=`
        <div class="brain-info-header" style="border-left:3px solid ${region.color};padding-left:12px">
          <h3 style="color:${region.color}">${region.name}</h3>
        </div>
        <p style="margin-top:12px">${region.desc}</p>
        <div style="margin-top:16px">
          <span style="font-size:11px;color:var(--text-dim);text-transform:uppercase;letter-spacing:1px">Connected Hormones</span>
          <div class="brain-hormones" style="margin-top:8px">
            ${region.hormones.map(h=>`<span class="brain-hormone-tag" style="background:${HORMONE_COLORS[h]}33;color:${HORMONE_COLORS[h]}">${h}</span>`).join('')}
          </div>
        </div>
        <div style="margin-top:16px;padding:12px;border-radius:8px;background:rgba(255,255,255,0.03);border:1px solid var(--border)">
          <span style="font-size:11px;color:var(--text-dim);text-transform:uppercase;letter-spacing:1px">Clinical Relevance</span>
          <p style="font-size:12px;color:var(--text-dim);margin-top:6px">${getRegionClinical(region.id)}</p>
        </div>
      `;
    });
    el.addEventListener('mouseenter',()=>{el.style.transform='translate(-50%,-50%) scale(1.15)';el.style.zIndex='10';});
    el.addEventListener('mouseleave',()=>{el.style.transform='translate(-50%,-50%) scale(1)';el.style.zIndex='1';});
  });

  // Drag to rotate
  const scene=document.getElementById('brain3d');
  scene.addEventListener('mousedown',e=>{brainDragging=true;brainLastX=e.clientX;brainLastY=e.clientY;brainAutoRotate=false;});
  window.addEventListener('mousemove',e=>{if(!brainDragging)return;brainRotY+=(e.clientX-brainLastX)*0.4;brainRotX+=(e.clientY-brainLastY)*0.4;brainRotX=Math.max(-60,Math.min(60,brainRotX));brainLastX=e.clientX;brainLastY=e.clientY;updateBrainRotation();});
  window.addEventListener('mouseup',()=>{brainDragging=false;setTimeout(()=>{brainAutoRotate=true;},2000);});

  updateBrainRotation();
  if(brainAutoRotate)autoRotateBrain();
}

function updateBrainRotation(){
  const stage=document.getElementById('brainStage');
  if(!stage)return;
  stage.style.transform=`rotateX(${brainRotX}deg) rotateY(${brainRotY}deg)`;
}

function autoRotateBrain(){
  function tick(){if(brainAutoRotate){brainRotY+=0.15;updateBrainRotation();}requestAnimationFrame(tick);}
  requestAnimationFrame(tick);
}

function getRegionClinical(id){
  const c={
    prefrontal:'Damage causes impulsivity, poor decision-making (Phineas Gage case). Dysfunction linked to ADHD, addiction, depression.',
    motor:'Stroke here causes contralateral paralysis. Neurodegeneration (ALS) leads to progressive motor loss.',
    somatosensory:'Damage causes numbness, inability to localize touch. Phantom limb pain involves reorganization.',
    auditory:'Damage causes cortical deafness. Auditory agnosia: inability to recognize sounds.',
    visual:'Damage causes cortical blindness, visual agnosia. V1 damage = loss of conscious vision.',
    amygdala:'Bilateral damage (Urbach-Wiethe) = inability to recognize fear. Hyperactivity linked to anxiety disorders.',
    hippocampus:'Damage causes anterograde amnesia (H.M. case). Critical for spatial navigation and memory consolidation.',
    hypothalamus:'Damage disrupts temperature, hunger, thirst, sleep, and endocrine regulation. Controls pituitary function.',
    thalamus:'Damage causes sensory loss, coma if bilateral. Acts as gateway for all sensory information.',
    pituitary:'Tumors (adenomas) cause hormone imbalances. Affects growth, reproduction, stress response, lactation.',
    brainstem:'Damage is often fatal. Controls vital functions: breathing, heart rate, consciousness.',
    cerebellum:'Damage causes ataxia, dysmetria, intention tremor. Also affects language and cognitive processing.',
    nacc:'Dysfunction linked to addiction, anhedonia, and motivational deficits. Central to reward circuitry.',
    cingulate:'Damage causes akinetic mutism. Dysfunction linked to chronic pain syndromes and depression.',
  };
  return c[id]||'Clinical significance varies by location and extent of involvement.';
}

// ===================== SAVE SIMULATION =====================
async function saveSim(sim){
  const saved=await api('/simulations',{method:'POST',body:JSON.stringify({type:sim.type||'scenario',title:sim.title,description:sim.description,icon:sim.icon,hormone:sim.hormone,profile:sim.profile})});
  if(saved && saved.error){
    console.warn('Save rejected:',saved.message);
    // show non-blocking toast in result area if possible
    return;
  }
  if(saved){const sims=await api('/simulations');if(sims && !sims.error){simulations=sims?.simulations||[];document.getElementById('totalSims').textContent=simulations.length;renderRecent();}}
}

// ===================== EEG MONITOR =====================
const EEG_CHANNELS=[
  {id:'Fp1',name:'Frontopolar 1',region:'Left front of the forehead',x:40,y:14,color:'#ef4444'},
  {id:'Fp2',name:'Frontopolar 2',region:'Right front of the forehead',x:60,y:14,color:'#f97316'},
  {id:'F3',name:'Frontal 3',region:'Left frontal region',x:32,y:32,color:'#6366f1'},
  {id:'F4',name:'Frontal 4',region:'Right frontal region',x:68,y:32,color:'#3b82f6'},
  {id:'C3',name:'Central 3',region:'Left central region',x:25,y:55,color:'#10b981'},
  {id:'C4',name:'Central 4',region:'Right central region',x:75,y:55,color:'#14b8a6'},
  {id:'P3',name:'Parietal 3',region:'Left parietal region',x:32,y:75,color:'#8b5cf6'},
  {id:'P4',name:'Parietal 4',region:'Right parietal region',x:68,y:75,color:'#a855f7'},
  {id:'O1',name:'Occipital 1',region:'Left back of the head',x:40,y:92,color:'#ec4899'},
  {id:'O2',name:'Occipital 2',region:'Right back of the head',x:60,y:92,color:'#d946ef'}
];

const EEG_BANDS={
  delta:{name:'Delta',range:'0.5-4 Hz',color:'#ef4444',freq:[0.5,4],desc:'Deep sleep, unconscious'},
  theta:{name:'Theta',range:'4-8 Hz',color:'#f97316',freq:[4,8],desc:'Drowsiness, light sleep, meditation'},
  alpha:{name:'Alpha',range:'8-13 Hz',color:'#10b981',freq:[8,13],desc:'Relaxed wakefulness, eyes closed'},
  beta:{name:'Beta',range:'13-30 Hz',color:'#3b82f6',freq:[13,30],desc:'Active thinking, focus, alertness'},
  gamma:{name:'Gamma',range:'30-45 Hz',color:'#8b5cf6',freq:[30,45],desc:'Higher cognition, peak concentration'}
};

const EEG_EMOTION_MAP={
  'Joy':{delta:0.1,theta:0.2,alpha:0.55,beta:0.45,gamma:0.2,channelMod:{F3:1.2,F4:1.1}},
  'Fear':{delta:0.08,theta:0.15,alpha:0.08,beta:0.8,gamma:0.35,channelMod:{Fp1:1.3,Fp2:1.3,Amygdala:1.4}},
  'Anger':{delta:0.06,theta:0.12,alpha:0.06,beta:0.85,gamma:0.4,channelMod:{Fp1:1.2,F3:1.3,C3:1.1}},
  'Sadness':{delta:0.2,theta:0.45,alpha:0.35,beta:0.15,gamma:0.06,channelMod:{Fp1:0.9,Fp2:0.9,O1:1.1}},
  'Surprise':{delta:0.05,theta:0.18,alpha:0.2,beta:0.65,gamma:0.3,channelMod:{Fp1:1.1,Fp2:1.1,F3:1.0}},
  'Disgust':{delta:0.1,theta:0.25,alpha:0.3,beta:0.45,gamma:0.12,channelMod:{Fp1:1.2,F3:0.9}},
  'Love':{delta:0.12,theta:0.3,alpha:0.6,beta:0.25,gamma:0.1,channelMod:{F3:1.1,F4:1.1,O1:1.0}},
  'Anxiety':{delta:0.06,theta:0.12,alpha:0.08,beta:0.82,gamma:0.38,channelMod:{Fp1:1.4,Fp2:1.4,F3:1.2,F4:1.2}},
  'Excitement':{delta:0.05,theta:0.15,alpha:0.18,beta:0.7,gamma:0.3,channelMod:{F3:1.2,F4:1.2,C3:1.1}},
  'Calm':{delta:0.2,theta:0.35,alpha:0.7,beta:0.12,gamma:0.06,channelMod:{O1:1.2,O2:1.2,P3:1.1}},
  'Guilt':{delta:0.12,theta:0.3,alpha:0.25,beta:0.45,gamma:0.1,channelMod:{Fp1:1.1,F3:1.0}},
  'Jealousy':{delta:0.08,theta:0.18,alpha:0.15,beta:0.65,gamma:0.2,channelMod:{Fp1:1.2,F3:1.1}},
  'Pride':{delta:0.08,theta:0.2,alpha:0.4,beta:0.5,gamma:0.15,channelMod:{F3:1.15,F4:1.1}},
  'Shame':{delta:0.15,theta:0.35,alpha:0.2,beta:0.4,gamma:0.08,channelMod:{Fp1:1.1,F3:0.9}},
  'Gratitude':{delta:0.12,theta:0.3,alpha:0.6,beta:0.2,gamma:0.08,channelMod:{F3:1.1,F4:1.1,O1:1.0}},
  'Hope':{delta:0.1,theta:0.25,alpha:0.5,beta:0.35,gamma:0.12,channelMod:{F3:1.15,F4:1.1}},
  'Loneliness':{delta:0.22,theta:0.4,alpha:0.3,beta:0.18,gamma:0.05,channelMod:{Fp1:0.9,O1:1.1}},
  'Frustration':{delta:0.06,theta:0.15,alpha:0.1,beta:0.75,gamma:0.3,channelMod:{Fp1:1.2,F3:1.2,C3:1.1}},
  'Relief':{delta:0.18,theta:0.3,alpha:0.65,beta:0.15,gamma:0.06,channelMod:{O1:1.1,O2:1.1,P3:1.0}},
  'Nostalgia':{delta:0.15,theta:0.4,alpha:0.5,beta:0.2,gamma:0.08,channelMod:{F3:1.1,O1:1.1}},
  'Empathy':{delta:0.1,theta:0.28,alpha:0.55,beta:0.3,gamma:0.1,channelMod:{F3:1.1,F4:1.1}},
  'Boredom':{delta:0.25,theta:0.4,alpha:0.45,beta:0.08,gamma:0.03,channelMod:{C3:0.8,C4:0.8}},
  'Contentment':{delta:0.15,theta:0.3,alpha:0.65,beta:0.15,gamma:0.06,channelMod:{O1:1.15,O2:1.15}},
  'Grief':{delta:0.2,theta:0.42,alpha:0.18,beta:0.25,gamma:0.06,channelMod:{Fp1:1.0,F3:0.85}}
};

const EEG_SCENARIO_KEYWORDS={
  'accident|crash|near.miss|danger|threat|attack':{delta:0.08,theta:0.15,alpha:0.08,beta:0.8,gamma:0.35,channelMod:{Fp1:1.3,Fp2:1.3,F3:1.1}},
  'speak|speech|audience|stage|present':{delta:0.06,theta:0.12,alpha:0.1,beta:0.75,gamma:0.3,channelMod:{Fp1:1.2,Fp2:1.2,F3:1.15}},
  'win|competition|victory|achieve|success':{delta:0.05,theta:0.15,alpha:0.2,beta:0.65,gamma:0.3,channelMod:{F3:1.2,F4:1.2}},
  'rescue|hero|save|brave':{delta:0.06,theta:0.18,alpha:0.15,beta:0.7,gamma:0.25,channelMod:{F3:1.1,F4:1.1,C3:1.1}},
  'gift|surprise|unexpected|present':{delta:0.08,theta:0.2,alpha:0.4,beta:0.5,gamma:0.15,channelMod:{F3:1.1,F4:1.1}},
  'spider|phobia|fear|scared|terrified':{delta:0.06,theta:0.12,alpha:0.06,beta:0.85,gamma:0.4,channelMod:{Fp1:1.4,Fp2:1.4}},
  'interview|job|exam|test|assessment':{delta:0.06,theta:0.1,alpha:0.1,beta:0.78,gamma:0.32,channelMod:{Fp1:1.2,F3:1.2}},
  'lose|lost|missing|gone':{delta:0.18,theta:0.38,alpha:0.3,beta:0.22,gamma:0.06,channelMod:{Fp1:1.0,O1:1.1}},
  'meditat|peace|calm|quiet|zen|mindful':{delta:0.22,theta:0.55,alpha:0.65,beta:0.08,gamma:0.05,channelMod:{O1:1.2,O2:1.2,P3:1.1}},
  'hug|reunite|partner|friend|love|hold':{delta:0.12,theta:0.3,alpha:0.6,beta:0.2,gamma:0.08,channelMod:{F3:1.1,F4:1.1}},
  'anxious|worry|nervous|stress|panic':{delta:0.06,theta:0.12,alpha:0.08,beta:0.82,gamma:0.38,channelMod:{Fp1:1.35,Fp2:1.35,F3:1.15}},
  'focus|concentrate|study|learn|think':{delta:0.05,theta:0.1,alpha:0.15,beta:0.72,gamma:0.3,channelMod:{F3:1.2,F4:1.2,C3:1.1}},
  'sleep|dream|rest|nap|tired':{delta:0.4,theta:0.5,alpha:0.25,beta:0.06,gamma:0.03,channelMod:{C3:0.8,C4:0.8,P3:0.9}},
  'exercise|run|gym|workout|sport|physical':{delta:0.04,theta:0.1,alpha:0.06,beta:0.72,gamma:0.28,channelMod:{C3:1.35,C4:1.35}},
  'music|listen|song|melody|rhythm':{delta:0.1,theta:0.3,alpha:0.5,beta:0.3,gamma:0.15,channelMod:{O1:1.2,O2:1.2,F3:1.0}},
  'cook|kitchen|recipe|food|eat':{delta:0.12,theta:0.25,alpha:0.45,beta:0.3,gamma:0.1,channelMod:{F3:1.05,C3:1.0}},
  'walk|nature|garden|forest|outside':{delta:0.15,theta:0.3,alpha:0.55,beta:0.2,gamma:0.08,channelMod:{O1:1.15,O2:1.15,P3:1.05}},
  'read|book|story|novel|study':{delta:0.08,theta:0.15,alpha:0.3,beta:0.6,gamma:0.2,channelMod:{F3:1.15,O1:1.1}},
  'argue|fight|conflict|dispute|yell':{delta:0.05,theta:0.1,alpha:0.05,beta:0.85,gamma:0.4,channelMod:{Fp1:1.3,F3:1.25,C3:1.15}}
};

let eegData=null;
let eegAnimFrame=null;
let eegAnimProgress=0;
let eegHover=null;
let highlightedEEGChannel=null;
let eegActiveBand='all';
let eegChannelSamples=320;
let eegCurrentScenario=null;
let eegLastProfile=null;

function initEEG(){
  renderEEGHeadmap();
}

function matchScenarioToEEG(scenario,emotion){
  if(emotion&&EEG_EMOTION_MAP[emotion])return{profile:EEG_EMOTION_MAP[emotion],title:emotion};
  if(scenario){
    const lower=scenario.toLowerCase();
    for(const[regex,profile]of Object.entries(EEG_SCENARIO_KEYWORDS)){
      if(new RegExp(regex,'i').test(lower))return{profile,title:scenario};
    }
  }
  if(emotion){
    const eLower=emotion.toLowerCase();
    for(const[name,profile]of Object.entries(EEG_EMOTION_MAP)){
      if(name.toLowerCase()===eLower)return{profile,title:name};
    }
  }
  return{profile:{delta:0.2,theta:0.25,alpha:0.5,beta:0.3,gamma:0.1},title:scenario||emotion||'Unknown'};
}

async function triggerEEGFromSimulation(scenario,emotion,icon){
  const match=matchScenarioToEEG(scenario,emotion);
  eegCurrentScenario=match.title;
  eegLastProfile=match.profile;
  eegAnimProgress=0;
  if(eegAnimFrame)cancelAnimationFrame(eegAnimFrame);
  eegAnimFrame=null;
  renderEEGBandBars(match.profile);
  updateEEGCurrentSim(match.title,icon||'\uD83E\uDDE0');
  renderEEGRecent();

  // Fetch real EEG data from backend dataset
  try{
    let url=`${API}/eeg/scenario?`;
    if(scenario)url+=`scenario=${encodeURIComponent(scenario)}`;
    if(emotion)url+=`&emotion=${encodeURIComponent(emotion)}`;
    const result=await fetch(url).then(r=>r.json());
    if(result&&result.data&&!result.error){
      eegData=result.data;
      eegChannelSamples=320;
      if(highlightedEEGChannel)selectEEGChannel(highlightedEEGChannel);
      if(currentPage==='eeg')animateEEGIn();
      return;
    }
  }catch(e){
    console.warn('Falling back to generated EEG:',e);
  }

  // Fallback: generate synthetic EEG if backend unavailable
  const result=generateEEGWaveform(match.profile);
  eegData=result.data;
  eegChannelSamples=result.nSamples;
  if(highlightedEEGChannel)selectEEGChannel(highlightedEEGChannel);
  if(currentPage==='eeg')animateEEGIn();
}

function updateEEGCurrentSim(title,icon){
  const el=document.getElementById('eegCurrentSim');
  if(!el)return;
  el.innerHTML=`<div class="eeg-sim-active"><span class="sim-icon">${icon}</span><div class="sim-info"><div class="sim-title">${title}</div><div class="sim-meta">EEG activity now</div></div></div>`;
}

function renderEEGRecent(){
  const list=document.getElementById('eegRecentList');
  if(!list)return;
  if(!simulations.length){list.innerHTML='<p class="empty-state">No simulations yet</p>';return;}
  list.innerHTML=simulations.slice(0,20).map(s=>{
    const isActive=s.title===eegCurrentScenario;
    return`<div class="eeg-recent-item${isActive?' active':''}" data-title="${(s.title||'').replace(/"/g,'&quot;')}" data-type="${s.type}" data-icon="${s.icon||'🧠'}">
      <div class="eeg-recent-icon">${s.icon||'🧠'}</div>
      <div class="eeg-recent-info"><strong>${s.title}</strong><span>${s.type||''}</span></div>
    </div>`;
  }).join('');
  list.querySelectorAll('.eeg-recent-item').forEach(item=>{
    item.addEventListener('click',()=>{
      const title=item.dataset.title;
      const type=item.dataset.type;
      const icon=item.dataset.icon;
      triggerEEGFromSimulation(type==='scenario'?title:null,type==='emotion'?title:null,icon);
    });
  });
}

function eegStartDrawWhenReady(){
  if(eegAnimFrame)cancelAnimationFrame(eegAnimFrame);
  eegAnimFrame=null;
  let lastAttempt=0;
  function attempt(){
    lastAttempt++;
    if(!eegData){
      if(lastAttempt<120)eegAnimFrame=requestAnimationFrame(attempt);
      return;
    }
    try{
      const canvas=document.getElementById('eegChart');
      if(!canvas||!canvas.parentElement){
        if(lastAttempt<120)eegAnimFrame=requestAnimationFrame(attempt);
        return;
      }
      const r=canvas.parentElement.getBoundingClientRect();
      if(r.width<50||r.height<50){
        if(lastAttempt<120)eegAnimFrame=requestAnimationFrame(attempt);
        return;
      }
      eegAnimProgress=1;
      drawEEGFrame(1,1);
      startEEGIdle();
    }catch(e){
      console.error('EEG draw error:',e);
      if(lastAttempt<120)eegAnimFrame=requestAnimationFrame(attempt);
    }
  }
  setTimeout(()=>{eegAnimFrame=requestAnimationFrame(attempt);},50);
}

function renderEEGHeadmap(){
  const map=document.getElementById('eegHeadmap');
  if(!map)return;
  map.innerHTML='';
  const seen=new Set();
  EEG_CHANNELS.forEach(ch=>{
    if(seen.has(ch.id))return;
    seen.add(ch.id);
    const el=document.createElement('div');
    el.className='eeg-electrode';
    el.dataset.id=ch.id;
    el.dataset.label=ch.id;
    el.style.left=ch.x+'%';
    el.style.top=ch.y+'%';
    el.style.background=`radial-gradient(circle,${ch.color}cc,${ch.color}66)`;
    el.style.boxShadow=`0 0 8px ${ch.color}55`;
    el.addEventListener('click',()=>selectEEGChannel(ch.id));
    el.addEventListener('mouseenter',()=>{el.style.boxShadow=`0 0 16px ${ch.color}`;});
    el.addEventListener('mouseleave',()=>{el.style.boxShadow=`0 0 8px ${ch.color}55`;});
    map.appendChild(el);
  });
}

function selectEEGChannel(id){
  const ch=EEG_CHANNELS.find(c=>c.id===id);
  if(!ch)return;
  highlightedEEGChannel=highlightedEEGChannel===id?null:id;
  document.querySelectorAll('.eeg-electrode').forEach(el=>{
    el.classList.toggle('active',el.dataset.id===highlightedEEGChannel);
  });
  const panel=document.getElementById('eegInfoPanel');
  if(highlightedEEGChannel&&eegData){
    const bandPowers=computeBandPowers(eegData[id]);
    panel.innerHTML=`
      <div class="eeg-channel-detail">
        <div class="ch-name" style="color:${ch.color}">${ch.id} — ${ch.name}</div>
        <div class="ch-region">${ch.region}</div>
        <svg class="ch-spark" viewBox="0 0 256 48" preserveAspectRatio="none">
          <polyline fill="none" stroke="${ch.color}" stroke-width="1.2" stroke-linejoin="round"
            points="${eegData[id].map((v,i)=>`${(i/(eegData[id].length-1))*256},${24-v*20}`).join(' ')}"/>
        </svg>
        <div class="ch-stats">
          <div class="ch-stat"><div class="ch-stat-label">Amplitude</div><div class="ch-stat-value" style="color:${ch.color}">${(Math.max(...eegData[id])*100).toFixed(0)} µV</div></div>
          <div class="ch-stat"><div class="ch-stat-label">Dominant</div><div class="ch-stat-value" style="color:${ch.color}">${getDominantBand(bandPowers)}</div></div>
          <div class="ch-stat"><div class="ch-stat-label">Alpha</div><div class="ch-stat-value">${(bandPowers.alpha*100).toFixed(0)}%</div></div>
          <div class="ch-stat"><div class="ch-stat-label">Beta</div><div class="ch-stat-value">${(bandPowers.beta*100).toFixed(0)}%</div></div>
        </div>
      </div>`;
  }else{
    panel.innerHTML=`<h3>Channel Info</h3><p style="color:var(--text-dim);font-size:13px;margin-top:8px">Click a channel on the head map or legend to inspect</p>`;
  }
  if(eegData)drawEEGFrame(1,eegAnimFrame?performance.now()/1000:0);
}

function getDominantBand(powers){
  let max=0,name='Alpha';
  Object.entries(powers).forEach(([band,p])=>{if(p>max){max=p;name=EEG_BANDS[band]?.name||band;}});
  return name;
}

function computeBandPowers(signal){
  const n=signal.length;
  const fft=new Float64Array(n);
  for(let i=0;i<n;i++)fft[i]=signal[i]-signal.reduce((a,b)=>a+b,0)/n;
  const powers={};
  Object.entries(EEG_BANDS).forEach(([band,{freq}])=>{
    let sum=0,count=0;
    for(let k=0;k<n/2;k++){
      const f=k*128/n;
      if(f>=freq[0]&&f<freq[1]){sum+=fft[k]*fft[k]+(fft[n-k]||0)*(fft[n-k]||0);count++;}
    }
    powers[band]=count>0?sum/count:0;
  });
  const total=Object.values(powers).reduce((a,b)=>a+b,0)||1;
  Object.keys(powers).forEach(b=>powers[b]/=total);
  return powers;
}

function generateEEGWaveform(profile){
  const bands=profile||{delta:0.2,theta:0.25,alpha:0.5,beta:0.3,gamma:0.1};
  const channelMod=bands.channelMod||{};
  const sampleRate=128;
  const duration=2;
  const nSamples=sampleRate*duration;
  const data={};
  EEG_CHANNELS.forEach(ch=>{
    const signal=new Float64Array(nSamples);
    const mod=channelMod[ch.id]||1;
    Object.entries(EEG_BANDS).forEach(([bandName,{freq}])=>{
      const amp=(bands[bandName]||0.1)*mod*0.35;
      const fLow=freq[0],fHigh=freq[1];
      const fCenter=(fLow+fHigh)/2;
      for(let i=0;i<nSamples;i++){
        const t=i/sampleRate;
        const envelope=Math.sin(Math.PI*t/duration);
        signal[i]+=amp*envelope*Math.sin(2*Math.PI*fCenter*t+Math.random()*0.3);
        if(fHigh>fLow+1){
          const f2=fLow+(fHigh-fLow)*0.6;
          signal[i]+=amp*0.3*envelope*Math.sin(2*Math.PI*f2*t+Math.random()*0.5);
        }
      }
    });
    for(let i=0;i<nSamples;i++)signal[i]+=((Math.random()-0.5)*0.02);
    const peak=Math.max(...Array.from(signal).map(Math.abs))||1;
    for(let i=0;i<nSamples;i++)signal[i]=signal[i]/peak*0.45+0.5;
    data[ch.id]=Array.from(signal);
  });
  return{data,sampleRate,nSamples,duration};
}

function animateEEGIn(){
  eegAnimProgress+=0.02;
  if(eegAnimProgress>=1){eegAnimProgress=1;drawEEGFrame(1);startEEGIdle();return;}
  drawEEGFrame(easeOutCubic(eegAnimProgress));
  eegAnimFrame=requestAnimationFrame(animateEEGIn);
}

function startEEGIdle(){
  if(eegAnimFrame)cancelAnimationFrame(eegAnimFrame);
  let t=0;
  function idle(){
    t+=0.02;
    if(!eegData){eegAnimFrame=requestAnimationFrame(idle);return;}
    const canvas=document.getElementById('eegChart');
    if(canvas){
      const rect=canvas.parentElement.getBoundingClientRect();
      if(rect.width<50||rect.height<50){eegAnimFrame=requestAnimationFrame(idle);return;}
    }
    drawEEGFrame(1,t);
    eegAnimFrame=requestAnimationFrame(idle);
  }
  eegAnimFrame=requestAnimationFrame(idle);
}

function drawEEGFrame(progress,idleTime=0){
  if(!eegData)return false;
  const canvas=document.getElementById('eegChart');
  if(!canvas)return false;
  const parent=canvas.parentElement;
  if(!parent)return false;
  const rect=parent.getBoundingClientRect();
  const W=rect.width,H=rect.height;
  if(W<50||H<50)return false;
  const ctx=canvas.getContext('2d');
  const dpr=window.devicePixelRatio||1;
  canvas.width=W*dpr;canvas.height=H*dpr;
  ctx.scale(dpr,dpr);
  const pad={top:12,right:16,bottom:28,left:44};
  const cW=W-pad.left-pad.right,cH=H-pad.top-pad.bottom;
  ctx.clearRect(0,0,W,H);

  const channels=Object.keys(eegData);
  const nCh=channels.length;
  const chH=cH/nCh;
  const drawCount=Math.floor(progress*eegChannelSamples);

  ctx.strokeStyle='rgba(100,100,200,0.06)';ctx.lineWidth=1;
  for(let i=0;i<=nCh;i++){
    const y=pad.top+chH*i;
    ctx.beginPath();ctx.moveTo(pad.left,y);ctx.lineTo(W-pad.right,y);ctx.stroke();
  }

  channels.forEach((chId,chIdx)=>{
    const ch=EEG_CHANNELS.find(c=>c.id===chId);
    if(!ch)return;
    const values=eegData[chId];
    const color=ch.color;
    const isHl=highlightedEEGChannel===chId;
    const isDim=highlightedEEGChannel&&!isHl;
    const alpha=isDim?0.15:1;
    const yCenter=pad.top+chH*chIdx+chH/2;
    const amplitude=chH*0.38;

    ctx.globalAlpha=alpha*0.4;
    ctx.font='9px JetBrains Mono,monospace';
    ctx.textAlign='right';
    ctx.fillStyle=color;
    ctx.fillText(chId,pad.left-6,yCenter+3);
    ctx.globalAlpha=alpha;

    const points=[];
    const step=Math.max(1,Math.floor(values.length/800));
    for(let i=0;i<drawCount&&i<values.length;i+=step){
      const x=pad.left+(cW/(values.length-1))*i;
      const w=Math.sin(idleTime*1.5+i*0.02+chIdx*0.4)*0.005;
      const v=values[i]+w;
      const y=yCenter-amplitude*(v-0.5)*2;
      points.push({x,y});
    }
    if(points.length<2)return false;

    ctx.beginPath();
    ctx.strokeStyle=color;
    ctx.lineWidth=isHl?2.5:1.2;
    ctx.lineJoin='round';
    ctx.lineCap='round';
    ctx.globalAlpha=alpha*(isHl?1:0.85);
    points.forEach((p,i)=>i===0?ctx.moveTo(p.x,p.y):ctx.lineTo(p.x,p.y));
    ctx.stroke();

    if(isHl){
      ctx.beginPath();
      ctx.strokeStyle=color;
      ctx.lineWidth=6;
      ctx.globalAlpha=0.15;
      points.forEach((p,i)=>i===0?ctx.moveTo(p.x,p.y):ctx.lineTo(p.x,p.y));
      ctx.stroke();
    }
    ctx.globalAlpha=1;
  });

  if(eegHover!==null){
    const hx=pad.left+(cW/(eegChannelSamples-1))*eegHover;
    ctx.beginPath();ctx.strokeStyle='rgba(148,163,184,0.25)';ctx.lineWidth=1;ctx.setLineDash([3,3]);
    ctx.moveTo(hx,pad.top);ctx.lineTo(hx,pad.top+cH);ctx.stroke();ctx.setLineDash([]);
    channels.forEach((chId,chIdx)=>{
      const values=eegData[chId];
      if(eegHover<values.length){
        const ch=EEG_CHANNELS.find(c=>c.id===chId);
        if(!ch)return;
        const v=values[eegHover];
        const yCenter=pad.top+chH*chIdx+chH/2;
        const amplitude=chH*0.38;
        const y=yCenter-amplitude*(v-0.5)*2;
        ctx.beginPath();ctx.arc(hx,y,3,0,Math.PI*2);ctx.fillStyle=ch.color;ctx.globalAlpha=highlightedEEGChannel===chId?1:0.6;ctx.fill();ctx.globalAlpha=1;
      }
    });
  }

  const legend=document.getElementById('eegLegend');
  if(legend){
    legend.innerHTML=EEG_CHANNELS.map(ch=>{
      const isH=highlightedEEGChannel===ch.id;
      const isD=highlightedEEGChannel&&!isH;
      return`<div class="legend-item${isH?' active':''}" data-eeg-channel="${ch.id}" style="opacity:${isD?0.3:1}"><span class="legend-dot" style="background:${ch.color};${isH?'box-shadow:0 0 8px '+ch.color:''}"></span>${ch.id}</div>`;
    }).join('');
    legend.querySelectorAll('.legend-item').forEach(item=>{
      item.addEventListener('mouseenter',()=>{highlightedEEGChannel=item.dataset.eegChannel;drawEEGFrame(1,eegAnimFrame?performance.now()/1000:0);});
      item.addEventListener('mouseleave',()=>{highlightedEEGChannel=null;drawEEGFrame(1,eegAnimFrame?performance.now()/1000:0);});
    });
  }
  return true;
}

const eegChartContainer=document.querySelector('.eeg-chart-container');
if(eegChartContainer){
  eegChartContainer.addEventListener('mousemove',e=>{
    if(!eegData)return;
    const canvas=document.getElementById('eegChart');
    const rect=canvas.getBoundingClientRect();
    const pad={left:44,right:16};
    const cW=rect.width-pad.left-pad.right;
    const ratio=(e.clientX-rect.left-pad.left)/cW;
    eegHover=Math.max(0,Math.min(eegChannelSamples-1,Math.round(ratio*(eegChannelSamples-1))));
    drawEEGFrame(1,eegAnimFrame?performance.now()/1000:0);
  });
  eegChartContainer.addEventListener('mouseleave',()=>{eegHover=null;drawEEGFrame(1,eegAnimFrame?performance.now()/1000:0);});
}

document.querySelectorAll('.eeg-band-controls .chart-btn').forEach(btn=>{
  btn.addEventListener('click',()=>{
    document.querySelectorAll('.eeg-band-controls .chart-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    eegActiveBand=btn.dataset.band;
    if(eegData)drawEEGFrame(1,eegAnimFrame?performance.now()/1000:0);
  });
});

function renderEEGBandBars(profile){
  const bars=document.getElementById('bandBars');
  if(!bars)return;
  const entries=Object.entries(EEG_BANDS);
  const maxVal=Math.max(...entries.map(([k])=>profile[k]||0),0.01);
  bars.innerHTML=entries.map(([key,{name,color}])=>{
    const val=profile[key]||0;
    const pct=(val/maxVal*100).toFixed(0);
    return`<div class="band-row"><span class="band-label" style="color:${color}">${name}</span><div class="band-bar-wrap"><div class="band-bar" style="width:${pct}%;background:${color}"></div></div><span class="band-pct">${(val*100).toFixed(0)}%</span></div>`;
  }).join('');
}

// ===================== INIT =====================
checkStatus();loadDashboard();initEmotions();initEEG();renderBrain();
