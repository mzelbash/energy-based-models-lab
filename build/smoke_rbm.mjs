import puppeteer from 'puppeteer-core';
const CHROME='C:/Program Files/Google/Chrome/Application/chrome.exe';
const b=await puppeteer.launch({executablePath:CHROME,headless:'new',args:['--no-sandbox','--use-gl=swiftshader','--ignore-gpu-blocklist']});
const p=await b.newPage(); const errs=[]; p.on('pageerror',e=>errs.push(e.message));
await p.goto('http://localhost:8000/',{waitUntil:'networkidle2',timeout:60000});
await p.waitForFunction(()=>{const t=document.querySelector('#model-status .txt');return t&&t.textContent.includes('loaded');},{timeout:45000}).catch(()=>{});
const out=await p.evaluate(async()=>{
  const r=await import('/assets/js/rbm.js');
  const data=await import('/assets/js/data.js');
  const tf=window.tf;
  const split=data.splitIndices();
  const batch=data.batchTensor(split.train.slice(0,300));
  const rbm=new r.RBM();
  const v=r.RBM.binarize(batch);
  const reconErr=()=>tf.tidy(()=>{const h=r.RBM.sample(rbm.hGivenV(v));const vp=rbm.vGivenH(h);return vp.sub(v).abs().mean().dataSync()[0];});
  const before=reconErr();
  await rbm.train(batch,{epochs:6,batchSize:64,lr:0.05});
  const after=reconErr();
  const leak0=tf.memory().numTensors;
  // gibbs loop leak check
  let vv=tf.keep(v.slice([0,0],[1,-1]));
  for(let i=0;i<20;i++){const g=rbm.gibbsOnce(vv);g.vProb.dispose();vv.dispose();vv=tf.keep(g.vSample);}
  vv.dispose();
  const leak1=tf.memory().numTensors;
  batch.dispose(); v.dispose(); rbm.dispose();
  return {before:before.toFixed(3), after:after.toFixed(3), gibbsLeak: leak1-leak0};
});
console.log('RBM',JSON.stringify(out)); console.log('ERRORS',errs.length,errs.join(';'));
await b.close(); if(errs.length)process.exit(2);
