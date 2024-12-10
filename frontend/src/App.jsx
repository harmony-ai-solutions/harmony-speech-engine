import {useState} from 'react'
import logo from './assets/images/harmony-link-icon-256.png';
import TTS from "./components/modules/TTS.jsx";

function App() {
    const [appName, setAppName] = useState('Harmony Speech Engine');
    const [appVersion, setAppVersion] = useState('v0.1.0-dev');

    const initialSettings = {
        "userapikey": "",
        "endpoint": "http://localhost:12080",
        "voiceconfigfile": ""
    };

    return (
        <div id="App">
            <div className="border-b-2 border-neutral-500">
                <ul className="flex cursor-pointer">
                    <li className="mr-1 h-10">
                        <img className="h-10 w-10" src={logo} id="logo" alt={appName}/>
                    </li>
                    <li className="mr-1 h-10">
                        <p className="inline-block py-2 px-4 text-orange-400 hover:text-orange-300 font-semibold">
                            <a href="https://project-harmony.ai/technology/" target="_blank">{appName} {appVersion}</a>
                        </p>
                    </li>
                </ul>
                <div className="bg-neutral-900 border-t-2 border-b border-neutral-500">
                    <TTS initialSettings={initialSettings}></TTS>
                </div>
                <div className="flex items-center justify-center bg-neutral-900">
                    <p className="py-1 px-2 text-orange-400">
                        <a href="https://project-harmony.ai/technology/" target="_blank">{appName} {appVersion} - &copy;2023-2024 Project Harmony.AI</a>
                    </p>
                </div>
            </div>

        </div>
    )
}

export default App
