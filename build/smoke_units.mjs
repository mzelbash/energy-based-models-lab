import puppeteer from 'puppeteer-core';
const CHROME='C:/Program Files/Google/Chrome/Application/chrome.exe';
const b=await puppeteer.launch({executablePath:CHROME,headless:'new',args:['--no-sandbox','--use-gl=swiftshader','--ignore-gpu-blocklist']});
const p=await b.newPage();
const errs=[]; p.on('pageerror',e=>errs.push(e.message));
await p.goto('http://localhost:8000/',{waitUntil:'networkidle2',timeout:60000});
await p.evaluate(()=>window.tf.ready());

const train=await p.evaluate(async()=>{
  const ec=await import('/assets/js/ebm-core.js');
  const md=await import('/assets/js/model-defs.js');
  const tf=window.tf;
  const before=tf.memory().numTensors;
  const model=md.buildEnergyModel(tf);
  const opt=tf.train.adam(1e-4);
  const buffer=ec.createBuffer(tf,model,{size:16,maxLen:64});
  const out=[];
  for(let i=0;i<3;i++){
    const real=tf.randomUniform([16,32,32,1],-1,1);
    const m=ec.cdTrainStep(tf,model,buffer,opt,real,{alpha:0.1,cdSteps:5,stepSize:10,noise:0.005});
    real.dispose(); out.push(m);
  }
  const fakes=buffer.currentSamples(4);
  const shape=fakes.shape.join('x'); fakes.dispose();
  buffer.dispose(); opt.dispose(); model.dispose();
  return {first:out[0], last:out[2], fakesShape:shape, leaked: tf.memory().numTensors-before};
});

const rbm=await p.evaluate(async()=>{
  const r=await import('/assets/js/rbm.js');
  const tf=window.tf;
  const before=tf.memory().numTensors;
  const rbm=new r.RBM();
  const batch=tf.randomUniform([32,32,32,1],-1,1);
  const v=r.RBM.binarize(batch);
  rbm.cdStep(v,0.05);
  const g=rbm.gibbsOnce(v);
  const shape=g.vSample.shape.join('x');
  g.vSample.dispose(); g.vProb.dispose(); v.dispose(); batch.dispose(); rbm.dispose();
  return {gibbsShape:shape, leaked: tf.memory().numTensors-before};
});

console.log('TRAIN',JSON.stringify(train));
console.log('RBM',JSON.stringify(rbm));
console.log('ERRORS',errs.length); errs.forEach(e=>console.log('  ',e));
await b.close();
if(errs.length)process.exit(2);
