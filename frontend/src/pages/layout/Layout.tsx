import { Outlet, Link } from "react-router-dom";
import styles from "./Layout.module.css";

const Layout = () => {
    return (
        <div className={styles.layout}>
            <header className={styles.header} role={"banner"}>
                <div className={styles.headerContainer}>
                    <Link to="/" className={styles.headerTitleContainer}>
                        <h3 className={styles.headerTitle}>{import.meta.env.VITE_APP_TITLE}</h3>
                    </Link>
                </div>
            </header>

            <Outlet />
        </div>
    );
};

export default Layout;
