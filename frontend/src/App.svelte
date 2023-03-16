<main>
  <div class="box" style="width:100px; width:{innerWidth*0.9}px; height:{innerHeight*0.9}px">
   

  </div>
</main>

<svelte:window bind:innerWidth bind:innerHeight />

<script>
  let innerWidth = 0
  let innerHeight = 0
  
  $: condition = innerWidth*1.33 <= innerHeight
</script>

<style>
  .box{
    background-color:cadetblue;

    position: absolute;
    top:0;
    bottom: 0;
    left: 0;
    right: 0;
    
    margin: auto;
  }
</style>

<!-- <script>
  import svelteLogo from './assets/svelte.svg'
  import Counter from './lib/Counter.svelte'
</script>

<main>
  <div>
    <a href="https://vitejs.dev" target="_blank" rel="noreferrer"> 
      <img src="/vite.svg" class="logo" alt="Vite Logo" />
    </a>
    <a href="https://svelte.dev" target="_blank" rel="noreferrer"> 
      <img src={svelteLogo} class="logo svelte" alt="Svelte Logo" />
    </a>
  </div>
  <h1>Vite + Svelte</h1>

  <div class="card">
    <Counter />
  </div>

  <p>
    Check out <a href="https://github.com/sveltejs/kit#readme" target="_blank" rel="noreferrer">SvelteKit</a>, the official Svelte app framework powered by Vite!
  </p>

  <p class="read-the-docs">
    Click on the Vite and Svelte logos to learn more
  </p>
</main>

<style>
  .logo {
    height: 6em;
    padding: 1.5em;
    will-change: filter;
    transition: filter 300ms;
  }
  .logo:hover {
    filter: drop-shadow(0 0 2em #646cffaa);
  }
  .logo.svelte:hover {
    filter: drop-shadow(0 0 2em #ff3e00aa);
  }
  .read-the-docs {
    color: #888;
  }
</style>
 -->
