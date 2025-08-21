const e = React.createElement;

function App(){
  const [file, setFile] = React.useState(null);
  const [sessionId, setSessionId] = React.useState(null);
  const [msgs, setMsgs] = React.useState([]);
  const [q, setQ] = React.useState('What is the termination clause?');
  const [loading, setLoading] = React.useState(false);
  const [summary, setSummary] = React.useState('');
  const [clauses, setClauses] = React.useState({});

  const backend = localStorage.getItem('backend_url') || 'http://localhost:8000';

  const onUpload = async () => {
    if(!file) return alert('Choose a PDF first');
    const fd = new FormData();
    fd.append('file', file);
    const res = await fetch(`${backend}/upload`, { method: 'POST', body: fd });
    const data = await res.json();
    if(data.session_id){
      setSessionId(data.session_id);
      setSummary(data.summary || '');
      setClauses(data.clauses || {});
      setMsgs([
        {role:'bot', text:`Uploaded. Pages: ${data.num_pages}. Chunks: ${data.num_chunks}. Ask me anything about this document.`}
      ]);
    }else{
      alert('Upload failed');
    }
  };

  const ask = async () => {
    if(!q.trim()) return;
    if(!sessionId) return alert('Upload a document first.');
    const question = q;
    setMsgs(prev => [...prev, {role:'user', text:question}]);
    setQ('');
    setLoading(true);
    try{
      const res = await fetch(`${backend}/chat`, {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({session_id: sessionId, message: question, top_k: 6})
      });
      const data = await res.json();
      const cites = (data.citations||[]).map(c => `[#${c.rank} p.${c.page}]`).join(' ');
      setMsgs(prev => [...prev, {role:'bot', text: (data.answer||'') + (cites ? `\n\nSources: ${cites}`:'') , citations: data.citations||[]}]);
    }catch(err){
      setMsgs(prev => [...prev, {role:'bot', text:'Error contacting backend.'}]);
    }finally{
      setLoading(false);
    }
  };

  return e('div',{className:'container'},
    e('div',{className:'header'},
      e('div',{className:'brand'},'PDF Chat RAG'),
      e('div',null,
        e('span',{className:'badge'},'RAG'),
        e('span',{className:'badge'},'OpenAI'),
        e('span',{className:'badge'},'FAISS')
      )
    ),
    e('div',{className:'row'},
      e('div',{className:'card'},
        e('div',{className:'upload'},
          e('input',{type:'file',accept:'application/pdf', onChange:e=>setFile(e.target.files[0])}),
          e('div',{className:'small'},'Upload a contract/book PDF (>=500 pages supported).'),
          e('button',{onClick:onUpload, style:{marginTop:8}},'Upload & Index')
        ),
        summary ? e('div',{className:'section'},
          e('div',null, e('span',{className:'badge'},'Summary')),
          e('div',{className:'summary'}, summary)
        ) : null,
        Object.keys(clauses||{}).length ? e('div',{className:'section'},
          e('div',null, e('span',{className:'badge'},'Detected Clauses')),
          e('div',null, Object.entries(clauses).map(([k,v]) => e('div',{key:k, className:'citation'}, e('b',null,k+': '), String(v))))
        ) : null
      ),
      e('div',{className:'card chat'},
        e('div',{className:'chatWindow'},
          msgs.map((m,i) => e('div',{key:i, className:`msg ${m.role==='user'?'user':'bot'}`},
            e('div',null,m.text),
            m.citations ? e('div',null, m.citations.slice(0,3).map(c=> e('div',{className:'citation',key:c.rank},
              `[#${c.rank}] page ${c.page} — ` + (c.excerpt||'').slice(0,160) + '…'
            ))) : null
          ))
        ),
        e('div',{className:'inputRow'},
          e('input',{type:'text',value:q,onChange:e=>setQ(e.target.value),placeholder:'Ask a question about the PDF…'}),
          e('button',{onClick:ask, disabled:loading}, loading ? 'Thinking…' : 'Ask')
        )
      )
    )
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(React.createElement(App));
